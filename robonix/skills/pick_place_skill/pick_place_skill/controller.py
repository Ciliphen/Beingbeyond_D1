#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Coordinate-based pick-and-place controller for the D1 arm + dexterous hand.

Given an absolute pick position and an offset relative to it, the controller
grasps at the pick point and releases at ``pick + offset``. Unlike
``block_grasp/grasp_controller.py`` there is **no vision**: no camera, no YOLO,
no hand-eye calibration and no head motion. Coordinates are supplied by the
caller, so the pipeline is just:

    open hand → approach above pick → descend → close → lift  (grasp)
    approach above place → descend → open → lift              (place)

It reuses the pure-compute stack the grasp pipeline already provides — the
``D1Kinematics`` FK/IK, the SLSQP solver in ``block_grasp.ik_scipy`` and the
motion/hand tunables in ``block_grasp.config`` — and drives the hardware through
injected primitive handles (``robot``, ``hand``). The small set of motion
helpers below is a trimmed version of the grasp controller's (no per-block OBB
wrist rotation, no workspace clamping, no detection).
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path

from block_grasp.ik_scipy import scipy_ik, scipy_ik_multi_restart
from block_grasp.config import (
    APPROACH_HEIGHT_OFFSET,
    ASIDE_POSITION,
    CATCH_DELAY_S,
    EE_PITCH_DEG,
    EE_ROLL_DEG,
    EE_YAW_DEG,
    GRASP_OK_MAX,
    GRASP_OK_MIN,
    GRAVITY_SAG_FACTOR,
    HAND_GRASP,
    HAND_OPEN,
    IK_FAIL_THRESHOLD,
    IK_MAX_ITERS,
    IK_N_RESTARTS,
    IK_POS_TOL,
    IK_TILT_TOL_DEG,
    IK_YAW_TOL_DEG,
    IK_Z_WEIGHT,
    INTERP_STEP_SIZE,
    JOINT_JUMP_THR_DEG,
    Z_SAFE,
)


class PosePickPlaceController:
    """Pick at an absolute point, place at ``pick + offset``. Synchronous,
    headless, no vision. Hardware is driven through injected primitive handles.
    """

    def __init__(
        self,
        *,
        robot,
        hand,
        urdf_path: str = "",
    ) -> None:
        """Set up kinematics and move the arm to a safe, table-perpendicular
        posture so the first approach has a well-conditioned IK start.

        Args:
            robot: Injected arm handle (HeadArmRobot-like) — the arm primitive.
            hand:  Injected hand handle (DexHand-like) — the hand primitive.
            urdf_path: Robot URDF; empty for the SDK default.
        """
        self._robot = robot
        self._hand = hand

        if not urdf_path:
            urdf_path = get_default_urdf_path()

        # Target EE orientation: roll+pitch keep the hand perpendicular to the
        # table, yaw is the default. No per-object rotation (no vision).
        self._R_target = R.from_euler(
            "xyz", [EE_ROLL_DEG, EE_PITCH_DEG, EE_YAW_DEG], degrees=True
        ).as_matrix()

        print("[Init] Setting up kinematics ...", flush=True)
        self._kin = D1Kinematics(D1KinematicsConfig(urdf_path=urdf_path))

        self._startup()
        print("[Init] Ready.", flush=True)

    # ── Startup posture ─────────────────────────────────────────────────────

    def _startup(self) -> None:
        """Safe posture → lift to safe Z → rotate EE to target RPY → open hand.

        Mirrors the block-grasp startup (minus the head-to-camera step, which
        this skill does not need) so IK starts from a good configuration
        instead of the folded home pose.
        """
        print("[Init] Moving to safe posture ...", flush=True)
        q_init = np.radians([0, 0, 0, -60, 60, 0, 0, 0])
        self._robot.set_positions(q_init)
        self._robot.wait_until_reached(q_init, active_joint_indices=range(8))
        time.sleep(0.3)

        # Lift to a safe height (SDK arm-only IK).
        print("[Init] Lifting to safe height ...", flush=True)
        q_full = np.asarray(self._robot.get_positions(), dtype=float)
        q_head, q_arm = self._kin.split_q(q_full)
        T_cur = self._kin.ee_in_base(q_head, q_arm)
        p_lift = T_cur[:3, 3].copy()
        p_lift[2] = Z_SAFE + 0.05
        T_lift = np.eye(4)
        T_lift[:3, :3] = T_cur[:3, :3]
        T_lift[:3, 3] = p_lift
        q_hs, q_as, err, _ = self._kin.ik_T_ee_with_arm_only(T_lift, q_head, q_arm)
        if err < 0.05:
            cmd = np.concatenate([q_hs, q_as])
            self._robot.set_positions(cmd)
            self._robot.wait_until_reached(cmd, active_joint_indices=range(2, 8))
            q_head, q_arm = self._kin.split_q(cmd)
        else:
            print(f"  ⚠ Lift IK error: {err:.3f}", flush=True)

        # SLERP-rotate to the target RPY (SDK arm-only IK per step).
        print("[Init] Rotating to target RPY ...", flush=True)
        T_cur = self._kin.ee_in_base(q_head, q_arm)
        p_cur = T_cur[:3, 3]
        q0 = R.from_matrix(T_cur[:3, :3]).as_quat()
        q1 = R.from_matrix(self._R_target).as_quat()
        if np.dot(q0, q1) < 0:  # shortest path (quaternion double-cover)
            q1 = -q1
        omega = float(np.arccos(np.clip(np.dot(q0, q1), -1.0, 1.0)))
        n_rot = max(1, int(math.ceil(omega * 2 / 0.05)))
        for i in range(n_rot):
            a = (i + 1) / n_rot
            if abs(omega) < 1e-10:
                qi = q0
            else:
                qi = (np.sin((1 - a) * omega) * q0 + np.sin(a * omega) * q1) / np.sin(omega)
            T_rt = np.eye(4)
            T_rt[:3, :3] = R.from_quat(qi).as_matrix()
            T_rt[:3, 3] = p_cur
            q_hs, q_as, err, _ = self._kin.ik_T_ee_with_arm_only(T_rt, q_head, q_arm)
            if err < 0.05:
                cmd = np.concatenate([q_hs, q_as])
                self._robot.set_positions(cmd)
                time.sleep(0.02)
                q_head, q_arm = self._kin.split_q(cmd)
            else:
                print(f"  ⚠ Rot IK err={err:.3f} at step {i + 1}/{n_rot}", flush=True)

        print("[Init] Opening hand ...", flush=True)
        self._hand.set_joint_pos(HAND_OPEN)
        time.sleep(0.3)

    # ── Motion helpers (trimmed from grasp_controller) ──────────────────────

    def _get_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        q_full = np.asarray(self._robot.get_positions(), dtype=float)
        return self._kin.split_q(q_full)

    def _z_sag(self, x: float, y: float) -> float:
        """Gravity-sag Z compensation: the arm droops the further it reaches,
        so raise the target Z to counteract it (cubic model, dz ∝ r³)."""
        dist = math.sqrt(x * x + y * y)
        return GRAVITY_SAG_FACTOR * dist ** 3

    def _make_target_pose(self, x: float, y: float, z: float) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self._R_target
        T[:3, 3] = [x, y, z]
        return T

    def _interpolate_and_move(
        self,
        T_target: np.ndarray,
        step_size: float = INTERP_STEP_SIZE,
        z_weight: float = IK_Z_WEIGHT,
    ) -> bool:
        """Move EE to target pose using SLSQP IK with linear interpolation."""
        q_head, q_arm = self._get_joint_state()
        T_cur = self._kin.ee_in_base(q_head, q_arm)
        p_start = T_cur[:3, 3].copy()
        # Work in the uncompensated (nominal) frame: strip the sag already baked
        # into the current pose so the per-step sag below never double-counts it.
        p_start[2] -= self._z_sag(p_start[0], p_start[1])
        p_target = T_target[:3, 3]
        R_target = T_target[:3, :3]

        dist = float(np.linalg.norm(p_target - p_start))
        n_steps = max(1, int(dist / step_size))

        cmd = None
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            interp = p_start + alpha * (p_target - p_start)
            interp[2] += self._z_sag(interp[0], interp[1])   # gravity-sag comp
            T_step = np.eye(4)
            T_step[:3, :3] = R_target
            T_step[:3, 3] = interp
            try:
                q_hs, q_as, err, _ = scipy_ik(
                    self._kin, T_step, q_head, q_arm,
                    z_weight=z_weight, pos_tol=IK_POS_TOL,
                    tilt_tol_deg=IK_TILT_TOL_DEG, yaw_tol_deg=IK_YAW_TOL_DEG,
                    max_iters=IK_MAX_ITERS,
                )
                if np.isnan(err) or err > IK_FAIL_THRESHOLD:
                    print(f"  [IK] SLSQP fail at step {i}/{n_steps}: err={err:.3f}", flush=True)
                    return False
                # Reject near-singularity joint jumps (excl. j6 wrist roll).
                dq_max = float(np.max(np.abs(q_as[:5] - q_arm[:5])))
                if dq_max > math.radians(JOINT_JUMP_THR_DEG):
                    print(f"  [IK] joint jump {math.degrees(dq_max):.0f}° at step "
                          f"{i}/{n_steps} (near singularity), aborting.", flush=True)
                    return False
                cmd = np.concatenate([q_hs, q_as])
                self._robot.set_positions(cmd)
                time.sleep(0.02)
                q_head, q_arm = self._kin.split_q(cmd)
            except Exception as e:  # noqa: BLE001
                print(f"  [IK] Step {i} error: {e}", flush=True)
                return False

        # Wait for the arm to physically settle at the final pose so the caller
        # (descend / grasp) doesn't read a lagging, far-from-target position.
        if cmd is not None:
            self._robot.wait_until_reached(cmd, active_joint_indices=range(2, 8))
        return True

    def _refine_and_move(self, T_target: np.ndarray, z_weight: float = IK_Z_WEIGHT) -> float:
        """Fine-positioning with SLSQP multi-restart IK. Returns the final
        position error (metres)."""
        q_head, q_arm = self._get_joint_state()
        T_target = T_target.copy()
        T_target[2, 3] += self._z_sag(T_target[0, 3], T_target[1, 3])
        try:
            q_hs, q_as, best_err, _ = scipy_ik_multi_restart(
                self._kin, T_target, q_head, q_arm,
                n_restarts=IK_N_RESTARTS, z_weight=z_weight,
                pos_tol=IK_POS_TOL, tilt_tol_deg=IK_TILT_TOL_DEG,
                yaw_tol_deg=IK_YAW_TOL_DEG,
            )
            # Reject a distant IK branch: the arm is already at target from the
            # interpolation phase, so keep it put and report the pose it holds.
            dq_max = float(np.max(np.abs(q_as[:5] - q_arm[:5])))
            if dq_max > math.radians(JOINT_JUMP_THR_DEG):
                T_cur = self._kin.ee_in_base(q_head, q_arm)
                cur_err = float(np.linalg.norm(T_cur[:3, 3] - T_target[:3, 3]))
                print(f"  [IK] refine joint jump {math.degrees(dq_max):.0f}° "
                      f"(near singularity), keeping current pose (err={cur_err:.4f}).",
                      flush=True)
                return cur_err
            cmd = np.concatenate([q_hs, q_as])
            self._robot.set_positions(cmd)
            self._robot.wait_until_reached(cmd, active_joint_indices=range(2, 8))
            return float(best_err)
        except Exception as e:  # noqa: BLE001
            print(f"  [IK] SLSQP refine error: {e}", flush=True)
            return 999.0

    def _move_to_pose(self, T_target: np.ndarray, refine: bool = True) -> bool:
        """Move EE to target pose: SLSQP interpolation + multi-restart refinement."""
        if not self._interpolate_and_move(T_target):
            return False
        if refine:
            err = self._refine_and_move(T_target)
            p = T_target[:3, 3]
            print(f"  → ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})  err={err:.4f}", flush=True)
            return err < 0.05
        return True

    # ── Grasp / place at explicit coordinates ───────────────────────────────

    def _grasp_at(self, x: float, y: float, z: float) -> bool:
        """open → approach above → descend → close → lift → check success."""
        z_approach = z + APPROACH_HEIGHT_OFFSET

        print("[Pick] Opening hand ...", flush=True)
        self._hand.set_joint_pos(HAND_OPEN)
        time.sleep(CATCH_DELAY_S)

        print(f"[Pick] Approaching above ({x:.3f}, {y:.3f}, {z_approach:.3f})", flush=True)
        T_approach = self._make_target_pose(x, y, z_approach)
        if not self._interpolate_and_move(T_approach):
            print("[Pick] SLSQP approach failed, trying multi-restart ...", flush=True)
            if self._refine_and_move(T_approach) > 0.05:
                print("[Pick] approach failed, aborting.", flush=True)
                return False
        time.sleep(CATCH_DELAY_S)

        print(f"[Pick] Descending to ({x:.3f}, {y:.3f}, {z:.3f})", flush=True)
        if not self._move_to_pose(self._make_target_pose(x, y, z), refine=True):
            print("[Pick] Descent failed, aborting.", flush=True)
            return False
        time.sleep(CATCH_DELAY_S)

        print("[Pick] Closing hand ...", flush=True)
        self._hand.set_joint_pos(HAND_GRASP)
        time.sleep(CATCH_DELAY_S)

        print("[Pick] Lifting ...", flush=True)
        if not self._interpolate_and_move(T_approach):
            print("[Pick] Lift failed (object may still be grasped).", flush=True)
        time.sleep(CATCH_DELAY_S)

        # Grasp success: average of the 4 main fingers in the expected band.
        current_pos = self._hand.read_joint_pos()
        active = [current_pos[i] for i in [0, 2, 3, 4]]  # thumb, idx, mid, ring
        avg_pos = sum(active) / len(active)
        grasp_ok = GRASP_OK_MIN <= avg_pos <= GRASP_OK_MAX
        print(f"[Pick] {'OK' if grasp_ok else 'FAILED'} "
              f"(avg 4-finger pos={avg_pos:.2f}, expected [{GRASP_OK_MIN:.1f}–{GRASP_OK_MAX:.1f}])",
              flush=True)
        return grasp_ok

    def _place_at(self, x: float, y: float, z: float) -> bool:
        """approach above → descend → open → lift."""
        z_approach = z + APPROACH_HEIGHT_OFFSET

        print(f"[Place] Approaching ({x:.3f}, {y:.3f}, {z_approach:.3f})", flush=True)
        T_approach = self._make_target_pose(x, y, z_approach)
        if not self._interpolate_and_move(T_approach):
            print("[Place] Approach failed.", flush=True)
            return False
        time.sleep(CATCH_DELAY_S)

        print(f"[Place] Descending to ({x:.3f}, {y:.3f}, {z:.3f})", flush=True)
        if not self._move_to_pose(self._make_target_pose(x, y, z), refine=True):
            print("[Place] Descent failed.", flush=True)
            return False
        time.sleep(CATCH_DELAY_S)

        print("[Place] Opening hand ...", flush=True)
        self._hand.set_joint_pos(HAND_OPEN)
        time.sleep(CATCH_DELAY_S)

        if not self._interpolate_and_move(T_approach):
            print("[Place] Lift failed.", flush=True)
            return False
        return True

    # ── Public API (used by the skill node) ─────────────────────────────────

    def pick_and_place(
        self,
        pick_xyz: Sequence[float],
        place_xyz: Sequence[float],
    ) -> Dict[str, object]:
        """Grasp at ``pick_xyz`` then release at ``place_xyz`` (both base-frame
        (x, y, z) in metres). Returns a JSON-serialisable result. If the grasp
        fails the place is skipped."""
        px, py, pz = (float(v) for v in pick_xyz)
        qx, qy, qz = (float(v) for v in place_xyz)

        if not self._grasp_at(px, py, pz):
            return {"ok": False, "grasped": False, "placed": False,
                    "pick": [round(px, 3), round(py, 3), round(pz, 3)],
                    "place": [round(qx, 3), round(qy, 3), round(qz, 3)],
                    "message": "grasp failed"}
        placed = self._place_at(qx, qy, qz)
        return {"ok": bool(placed), "grasped": True, "placed": bool(placed),
                "pick": [round(px, 3), round(py, 3), round(pz, 3)],
                "place": [round(qx, 3), round(qy, 3), round(qz, 3)],
                "message": "picked and placed" if placed else "grasped but place failed"}

    def move_home(self) -> None:
        """Open the hand and park the arm at a safe pose (clears the workspace)."""
        self._hand.set_joint_pos(HAND_OPEN)
        time.sleep(CATCH_DELAY_S)
        ax, ay, az = ASIDE_POSITION
        print(f"[Home] Parking at ({ax:.3f}, {ay:.3f}, {az:.3f})", flush=True)
        try:
            self._interpolate_and_move(self._make_target_pose(ax, ay, az))
        except Exception as e:  # noqa: BLE001
            print(f"[Home] Failed: {e}", flush=True)

    def shutdown(self) -> None:
        """Safe release on the way out; hardware is owned by the primitives."""
        print("[Shutdown] Opening hand ...", flush=True)
        try:
            self._hand.set_joint_pos(HAND_OPEN)
            time.sleep(0.3)
        except Exception:  # noqa: BLE001
            pass
        for closer in (getattr(self._hand, "close_can", None),
                       getattr(self._robot, "close", None)):
            try:
                if closer is not None:
                    closer()
            except Exception:  # noqa: BLE001
                pass
        print("[Shutdown] Done.", flush=True)
