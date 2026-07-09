#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
Block grasp controller for the D1 dexterous hand.

Integrates the three-layer architecture:

    Perception  →  YOLO OBB detection + RealSense camera
    Control     →  HeadArmRobot (IK) + DexHand (fingers)
    Planning    →  detect → transform → plan → execute grasp/place

Mirrors roboarm's ``chess/catch_and_place.py`` + ``arm/arm_base.py``.

Uses **2D homography** (``pixel2pos``) for pixel→world transform — same as
roboarm, reusing the existing ``handeye_calib.npz``.  No depth camera needed.
Custom IK solvers (Jacobian + SLSQP) for Z-plane consistency.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# Project paths — allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clients.camera import D1CameraPrimitive
from object_detect.detect import (
    detect_objects_in_frame,
    draw_box,
    load_model,
)
from beingbeyond_d1_sdk.head_arm import HeadArmRobot
from beingbeyond_d1_sdk.dex_hand import DexHand
from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path

from block_grasp.ik_scipy import scipy_ik, scipy_ik_multi_restart

from .config import (
    APPROACH_HEIGHT_OFFSET,
    BLOCK_SIZE,
    CALIB_CAM_HEIGHT,
    CALIB_CAM_WIDTH,
    CALIB_PATH,
    CATCH_DELAY_S,
    CONF_THRESHOLD,
    DEFAULT_PLACE_Z,
    EE_PITCH_DEG,
    EE_ROLL_DEG,
    EE_YAW_DEG,
    GRASP_OFFSET_X,
    GRASP_OFFSET_Y,
    GRASP_OK_MAX,
    GRASP_OK_MIN,
    GRASP_YAW_OFFSET_DEG,
    GRASP_Z_OFFSET,
    ASIDE_POSITION,
    GRAVITY_SAG_FACTOR,
    HAND_GRASP,
    HAND_OPEN,
    PLACE_DISTANCE_THRESHOLD,
    STACK_ENABLED,
    STACK_POSITION,
    HEAD_PITCH_DEG,
    HEAD_YAW_DEG,
    IK_FAIL_THRESHOLD,
    IK_MAX_ITERS,
    IK_N_RESTARTS,
    IK_POS_TOL,
    IK_TILT_TOL_DEG,
    IK_YAW_TOL_DEG,
    IK_Z_WEIGHT,
    INTERP_STEP_SIZE,
    IOU_THRESHOLD,
    JOINT_JUMP_THR_DEG,
    MAX_DXY,
    OBB_GRASP_RATIO,
    PLACE_POSITIONS,
    Z_SAFE,
)
from .coordinate_utils import (
    estimate_grasp_angle_deg,
    obb_bottom_center,
    pixel_to_world_2d,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BlockDetection:
    """A single detected block with its world position resolved via homography."""

    class_name: str
    class_id: int
    score: float
    # Pixel-space OBB centre (for draw_box)
    u: float
    v: float
    w: float
    h: float
    r_rad: float
    # Pixel-space bottom-centre of OBB (for world-coordinate lookup)
    u_bot: float
    v_bot: float
    # World-space (base frame) — XY from bottom-centre via homography
    x: float
    y: float
    z: float
    # Recommended EE yaw offset (degrees) from OBB long edge
    grasp_angle_deg: float


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class BlockGraspController:
    """Top-level controller for YOLO-guided block grasping with D1.

    Usage::

        ctrl = BlockGraspController(
            model_path="object_detect/runs/train-3/weights/best.pt",
        )
        ctrl.run_loop()
    """

    def __init__(
        self,
        model_path: str,
        hand_type: str = "right",
        hand_can: str = "can0",
        arm_dev: str = "/dev/ttyUSB0",
        arm_baud: int = 1_000_000,
        urdf_path: str = "",
        cam_width: int = 640,
        cam_height: int = 480,
        cam_fps: int = 30,
        device: str = "",
        headless: bool = False,
        auto_grasp: bool = False,
    ) -> None:
        """Initialise all hardware, models, and kinematics.

        Args:
            model_path: Path to YOLO ``.pt`` checkpoint.
            hand_type:  ``"right"`` or ``"left"``.
            hand_can:   CAN interface name (e.g. ``"can0"``).
            arm_dev:    Serial device for the head–arm chain.
            arm_baud:   Serial baudrate.
            urdf_path:  Robot URDF; empty for SDK default.
            cam_width:  Camera image width.
            cam_height: Camera image height.
            cam_fps:    Camera frame rate.
            device:     Torch device for YOLO (empty = auto).
            headless:   If True, skip OpenCV display windows.
            auto_grasp: If True, automatically grasp detected blocks.
                        If False, press SPACE to trigger grasp.
        """
        self._headless = headless
        self._auto_grasp = auto_grasp

        # ── URDF path ────────────────────────────────────────────────────
        if not urdf_path:
            urdf_path = get_default_urdf_path()

        # ── Calibration ──────────────────────────────────────────────────
        print(f"[Init] Loading calibration: {CALIB_PATH}")
        if not os.path.isfile(CALIB_PATH):
            raise FileNotFoundError(
                f"Calibration file not found: {CALIB_PATH}\n"
                f"Run block_grasp/calibrate_handeye.py first."
            )
        calib = np.load(CALIB_PATH, allow_pickle=True)
        self._H: np.ndarray = calib["H"]                 # 3×3 homography
        self._calib_head_yaw: float = float(calib["head_yaw"])
        self._calib_head_pitch: float = float(calib["head_pitch"])
        # Table height from calibration points
        if "world_pts" in calib:
            self._W: Optional[np.ndarray] = calib["world_pts"]  # (N, 3)
            self._z_table: float = float(np.median(self._W[:, 2]))
        else:
            self._W = None
            self._z_table: float = 0.08  # fallback
        print(f"       head: yaw={math.degrees(self._calib_head_yaw):.0f}°  "
              f"pitch={math.degrees(self._calib_head_pitch):.0f}°  "
              f"z_table={self._z_table:.3f}")

        # ── Perception ─────────────────────────────────────────────────
        print("[Init] Opening RealSense camera ...")
        self._camera = D1CameraPrimitive(
            width=cam_width, height=cam_height, fps=cam_fps
        )

        print(f"[Init] Loading YOLO model from {model_path} ...")
        self._model = load_model(model_path, device=device)

        # ── Control: arm + head ─────────────────────────────────────────
        print(f"[Init] Opening head–arm on {arm_dev} ...")
        self._robot = HeadArmRobot(
            urdf_path=urdf_path, dev=arm_dev, baudrate=arm_baud
        )

        # ── Control: dexterous hand ─────────────────────────────────────
        print(f"[Init] Opening dexterous hand on {hand_can} ({hand_type}) ...")
        self._hand = DexHand(
            hand_type=hand_type, can_iface=hand_can, baudrate=1_000_000
        )

        # ── Kinematics ──────────────────────────────────────────────────
        print("[Init] Setting up kinematics ...")
        self._kin = D1Kinematics(D1KinematicsConfig(urdf_path=urdf_path))

        # ── Startup: safe posture → set head → lift → rotate → open hand ──

        # Target EE orientation: roll+pitch ⟂ table, yaw is default.
        # Per-block OBB angle is applied as a delta Z-rotation on top.
        self._R_target = R.from_euler(
            'xyz',
            [EE_ROLL_DEG, EE_PITCH_DEG, EE_YAW_DEG],
            degrees=True,
        ).as_matrix()

        print("[Init] Moving to safe posture ...")
        q_init = np.radians([0, 0, 0, -60, 60, 0, 0, 0])
        self._robot.set_positions(q_init)
        self._robot.wait_until_reached(q_init, active_joint_indices=range(8))
        time.sleep(0.3)

        print(f"[Init] Setting head to calibration position "
              f"(yaw={HEAD_YAW_DEG:.0f}°, pitch={HEAD_PITCH_DEG:.0f}°) ...")
        q = np.asarray(self._robot.get_positions(), dtype=float)
        q[0] = math.radians(HEAD_YAW_DEG)
        q[1] = math.radians(HEAD_PITCH_DEG)
        self._robot.set_positions(q)
        self._robot.wait_until_reached(q, active_joint_indices=[0, 1])
        time.sleep(0.3)

        # ── Lift to safe Z (SDK IK — same as test_click_goto.py) ────────
        print("[Init] Lifting to safe height ...")
        q_full = np.asarray(self._robot.get_positions(), dtype=float)
        q_head, q_arm = self._kin.split_q(q_full)
        T_cur = self._kin.ee_in_base(q_head, q_arm)
        p_lift = T_cur[:3, 3].copy()
        p_lift[2] = Z_SAFE + 0.05
        T_lift = np.eye(4)
        T_lift[:3, :3] = T_cur[:3, :3]
        T_lift[:3, 3] = p_lift
        q_hs, q_as, err, _ = self._kin.ik_T_ee_with_arm_only(
            T_lift, q_head, q_arm,
        )
        if err < 0.05:
            cmd = np.concatenate([q_hs, q_as])
            cmd[0] = math.radians(HEAD_YAW_DEG)
            cmd[1] = math.radians(HEAD_PITCH_DEG)
            self._robot.set_positions(cmd)
            self._robot.wait_until_reached(cmd, active_joint_indices=range(2, 8))
            q_head, q_arm = self._kin.split_q(cmd)
        else:
            print(f"  ⚠ Lift IK error: {err:.3f}")

        # ── SLERP rotate to target RPY (SDK IK — same as test_click_goto.py) ──
        print("[Init] Rotating to target RPY ...")
        T_cur = self._kin.ee_in_base(q_head, q_arm)
        p_cur = T_cur[:3, 3]
        R_cur = T_cur[:3, :3]
        q0 = R.from_matrix(R_cur).as_quat()
        q1 = R.from_matrix(self._R_target).as_quat()
        # Shortest path (quaternion double-cover)
        if np.dot(q0, q1) < 0:
            q1 = -q1
        omega = float(np.arccos(np.clip(np.dot(q0, q1), -1.0, 1.0)))
        angle = omega * 2
        n_rot = max(1, int(math.ceil(angle / 0.05)))
        for i in range(n_rot):
            a = (i + 1) / n_rot
            if abs(omega) < 1e-10:
                qi = q0
            else:
                qi = (np.sin((1 - a) * omega) * q0 +
                      np.sin(a * omega) * q1) / np.sin(omega)
            Ri = R.from_quat(qi).as_matrix()
            T_rt = np.eye(4)
            T_rt[:3, :3] = Ri
            T_rt[:3, 3] = p_cur
            q_hs, q_as, err, _ = self._kin.ik_T_ee_with_arm_only(
                T_rt, q_head, q_arm,
            )
            if err < 0.05:
                cmd = np.concatenate([q_hs, q_as])
                cmd[0] = math.radians(HEAD_YAW_DEG)
                cmd[1] = math.radians(HEAD_PITCH_DEG)
                self._robot.set_positions(cmd)
                time.sleep(0.02)
                q_head, q_arm = self._kin.split_q(cmd)
            else:
                print(f"  ⚠ Rot IK err={err:.3f} at step {i+1}/{n_rot}")
        rpy = R.from_matrix(self._R_target).as_euler('xyz', degrees=True)
        print(f"       RPY=({rpy[0]:.0f}, {rpy[1]:.0f}, {rpy[2]:.0f})")

        # Open hand
        print("[Init] Opening hand ...")
        self._hand.set_joint_pos(HAND_OPEN)
        time.sleep(0.3)

        # ── State ───────────────────────────────────────────────────────
        self._busy = False
        self._step_count = 0
        self._stack_count = 0  # blocks already stacked at STACK_POSITION

        # Workspace reference: current EE XY after startup (for clamping detections)
        T_start = self._kin.ee_in_base(q_head, q_arm)
        self._ws_x0 = float(T_start[0, 3])
        self._ws_y0 = float(T_start[1, 3])

        print("[Init] Ready.")

    # ── Perception helpers ─────────────────────────────────────────────────

    def _get_table_z(self, x: float, y: float) -> float:
        """Interpolate table Z at (x, y) from calibration points.

        Uses inverse-distance-weighted average of the 3 nearest calibration
        points, same as ``test_click_goto.py``.  Handles tilted tables.
        """
        if self._W is None or len(self._W) < 3:
            return self._z_table
        dists = np.sqrt((self._W[:, 0] - x) ** 2 + (self._W[:, 1] - y) ** 2)
        idx = np.argsort(dists)[:3]
        if dists[idx[0]] < 1e-6:
            return float(self._W[idx[0], 2])
        wgt = 1.0 / (dists[idx] + 0.001)
        wgt /= wgt.sum()
        return float(np.dot(wgt, self._W[idx, 2]))

    def detect_blocks(self, frame: np.ndarray) -> List[BlockDetection]:
        """Detect blocks in a colour frame and resolve to world coordinates.

        Pipeline:
        1. YOLO OBB → pixel bounding box
        2. Homography → table-plane XY (parallax-skewed for objects above table)
        3. Perspective correction → true XY accounting for cube height

        Uses camera pose from FK, table height from calibration, and
        known cube height (5 cm).  No depth camera needed.

        Args:
            frame: RGB colour image (H, W, 3).

        Returns:
            List of resolved ``BlockDetection``, sorted by score descending.
        """
        detections = detect_objects_in_frame(
            self._model, frame, conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD
        )

        # Camera position in base frame (for perspective correction)
        q_full = np.asarray(self._robot.get_positions(), dtype=float)
        q_head, q_arm = self._kin.split_q(q_full)
        T_base_cam = self._kin.camera_in_base(q_head, q_arm)
        cx, cy, cz = T_base_cam[:3, 3]  # camera origin in base frame

        # Scale factors: camera resolution → calibration resolution
        sx = CALIB_CAM_WIDTH / max(frame.shape[1], 1)
        sy = CALIB_CAM_HEIGHT / max(frame.shape[0], 1)

        blocks: List[BlockDetection] = []
        for (u, v, w, h, r), score, cls_id, cls_name in detections:
            # Bottom-centre pixel of the OBB (see coordinate_utils)
            u_bot, v_bot = obb_bottom_center(
                u, v, w, h, np.rad2deg(r), ratio=OBB_GRASP_RATIO,
            )

            # Homography: pixel → table-plane XY (first pass)
            u_calib = u_bot * sx
            v_calib = v_bot * sy
            wx_hom, wy_hom = pixel_to_world_2d(u_calib, v_calib, self._H)

            # Interpolate table Z at this XY (handles tilted tables)
            z_tbl = self._get_table_z(wx_hom, wy_hom)
            z_obj = z_tbl + BLOCK_SIZE / 2.0
            denom = z_tbl - cz
            t_corr = (z_obj - cz) / denom if abs(denom) > 0.001 else 1.0

            # Perspective correction with local table Z
            wx = cx + (wx_hom - cx) * t_corr
            wy = cy + (wy_hom - cy) * t_corr

            # Clamp to workspace
            wx = float(np.clip(wx, self._ws_x0 - MAX_DXY, self._ws_x0 + MAX_DXY))
            wy = float(np.clip(wy, self._ws_y0 - MAX_DXY, self._ws_y0 + MAX_DXY))

            # Z: top surface of cube on table
            z_top = z_tbl + BLOCK_SIZE

            grasp_angle_deg = estimate_grasp_angle_deg(
                u, v, w, h, np.rad2deg(r)
            )

            blocks.append(BlockDetection(
                class_name=cls_name,
                class_id=cls_id,
                score=score,
                u=u, v=v, w=w, h=h, r_rad=r,
                u_bot=u_bot, v_bot=v_bot,
                x=wx, y=wy, z=z_top,
                grasp_angle_deg=grasp_angle_deg,
            ))

        blocks.sort(key=lambda b: b.score, reverse=True)
        return blocks

    # ── Motion primitives ──────────────────────────────────────────────────

    def _get_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current (q_head, q_arm) from the robot."""
        q_full = np.asarray(self._robot.get_positions(), dtype=float)
        return self._kin.split_q(q_full)

    def _z_sag(self, x: float, y: float) -> float:
        """Gravity-sag Z compensation at horizontal distance from base.

        The arm droops under its own weight the further it reaches, so raise
        the target Z to counteract it.  Cubic model (dz ∝ r³), same as teleop.
        """
        dist = math.sqrt(x * x + y * y)
        return GRAVITY_SAG_FACTOR * dist ** 3

    def _make_target_pose(
        self,
        x: float,
        y: float,
        z: float,
    ) -> np.ndarray:
        """Build a 4×4 target pose in base frame (base RPY, no yaw offset).

        The IK solves for this base pose.  Wrist rotation (OBB angle) is
        applied directly to **joint_6** after IK — much simpler and does
        not affect position accuracy.

        Args:
            x, y, z:  Target position in base frame (metres).

        Returns:
            4×4 homogeneous transform.
        """
        T = np.eye(4)
        T[:3, :3] = self._R_target
        T[:3, 3] = [x, y, z]
        return T

    def _interpolate_and_move(
        self,
        T_target: np.ndarray,
        step_size: float = INTERP_STEP_SIZE,
        z_weight: float = IK_Z_WEIGHT,
        j6_offset_rad: float = 0.0,
    ) -> bool:
        """Move EE to target pose using SLSQP IK with linear interpolation.

        *j6_offset_rad* is added to joint_6 after each IK step so the
        wrist rotates to the OBB angle without affecting position accuracy.

        Returns True on success.
        """
        q_head, q_arm = self._get_joint_state()
        T_cur = self._kin.ee_in_base(q_head, q_arm)
        p_start = T_cur[:3, 3].copy()
        # Work in the *uncompensated* (nominal) frame: strip the sag already
        # baked into the current pose so the per-step sag below never double-
        # counts it (otherwise the EE lurches up at the start of each move).
        p_start[2] -= self._z_sag(p_start[0], p_start[1])
        p_target = T_target[:3, 3]
        R_target = T_target[:3, :3]

        dist = float(np.linalg.norm(p_target - p_start))
        n_steps = max(1, int(dist / step_size))

        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            interp = p_start + alpha * (p_target - p_start)
            interp[2] += self._z_sag(interp[0], interp[1])   # gravity-sag comp
            T_step = np.eye(4)
            T_step[:3, :3] = R_target
            T_step[:3, 3] = interp

            try:
                q_hs, q_as, err, it = scipy_ik(
                    self._kin, T_step, q_head, q_arm,
                    z_weight=z_weight,
                    pos_tol=IK_POS_TOL, tilt_tol_deg=IK_TILT_TOL_DEG,
                    yaw_tol_deg=IK_YAW_TOL_DEG, max_iters=IK_MAX_ITERS,
                )
                if np.isnan(err) or err > IK_FAIL_THRESHOLD:
                    print(f"  [IK] SLSQP fail at step {i}/{n_steps}: "
                          f"err={err:.3f}  target=({interp[0]:.3f},{interp[1]:.3f},{interp[2]:.3f})")
                    return False

                # Reject near-singularity joint jumps (excl. j6 wrist roll)
                dq_max = float(np.max(np.abs(q_as[:5] - q_arm[:5])))
                if dq_max > math.radians(JOINT_JUMP_THR_DEG):
                    print(f"  [IK] joint jump {math.degrees(dq_max):.0f}° at step "
                          f"{i}/{n_steps} (near singularity), aborting.")
                    return False

                # Apply OBB yaw directly to joint_6 (wrist rotation)
                q_as[5] += j6_offset_rad
                # Normalise to shortest path from current joint_6
                diff = q_as[5] - q_arm[5]
                q_as[5] = q_arm[5] + (diff + math.pi) % (2 * math.pi) - math.pi

                cmd = np.concatenate([q_hs, q_as])
                cmd[0] = math.radians(HEAD_YAW_DEG)
                cmd[1] = math.radians(HEAD_PITCH_DEG)
                self._robot.set_positions(cmd)
                time.sleep(0.02)
                q_head, q_arm = self._kin.split_q(cmd)
            except Exception as e:
                print(f"  [IK] Step {i} error: {e}")
                return False

        return True

    def _refine_and_move(
        self,
        T_target: np.ndarray,
        z_weight: float = IK_Z_WEIGHT,
        j6_offset_rad: float = 0.0,
    ) -> float:
        """Fine-positioning with SLSQP multi-restart IK.

        *j6_offset_rad* is added to joint_6 after IK (OBB wrist rotation).

        Returns the final position error (metres).
        """
        q_head, q_arm = self._get_joint_state()

        # Gravity-sag compensation on the target Z (copy: caller reuses T_target)
        T_target = T_target.copy()
        T_target[2, 3] += self._z_sag(T_target[0, 3], T_target[1, 3])

        try:
            q_hs, q_as, best_err, _ = scipy_ik_multi_restart(
                self._kin, T_target, q_head, q_arm,
                n_restarts=IK_N_RESTARTS, z_weight=z_weight,
                pos_tol=IK_POS_TOL, tilt_tol_deg=IK_TILT_TOL_DEG,
                yaw_tol_deg=IK_YAW_TOL_DEG,
            )
            # Reject near-singularity joint jumps (excl. j6 wrist roll)
            dq_max = float(np.max(np.abs(q_as[:5] - q_arm[:5])))
            if dq_max > math.radians(JOINT_JUMP_THR_DEG):
                print(f"  [IK] refine joint jump {math.degrees(dq_max):.0f}° "
                      f"(near singularity), not moving.")
                return 999.0
            # Apply OBB yaw directly to joint_6 (wrist rotation)
            q_as[5] += j6_offset_rad
            # Normalise to shortest path from current joint_6
            diff = q_as[5] - q_arm[5]
            q_as[5] = q_arm[5] + (diff + math.pi) % (2 * math.pi) - math.pi

            cmd = np.concatenate([q_hs, q_as])
            cmd[0] = math.radians(HEAD_YAW_DEG)
            cmd[1] = math.radians(HEAD_PITCH_DEG)
            self._robot.set_positions(cmd)
            time.sleep(0.05)
            return float(best_err)
        except Exception as e:
            print(f"  [IK] SLSQP refine error: {e}")
            return 999.0

    def _move_to_pose(
        self,
        T_target: np.ndarray,
        refine: bool = True,
        j6_offset_rad: float = 0.0,
    ) -> bool:
        """Move EE to target pose: SLSQP interpolation + multi-restart refinement.

        Args:
            T_target:      4×4 target pose in base frame.
            refine:        If True, run SLSQP multi-restart refinement at the end.
            j6_offset_rad: Wrist rotation (joint_6) offset in radians.

        Returns True on success.
        """
        if not self._interpolate_and_move(T_target, j6_offset_rad=j6_offset_rad):
            return False

        if refine:
            err = self._refine_and_move(T_target, j6_offset_rad=j6_offset_rad)
            p_tgt = T_target[:3, 3]
            print(f"  → ({p_tgt[0]:.3f}, {p_tgt[1]:.3f}, {p_tgt[2]:.3f})  "
                  f"err={err:.4f}")
            return err < 0.05

        return True

    # ── Hand control ───────────────────────────────────────────────────────

    def _hand_open(self) -> None:
        """Open the dexterous hand."""
        self._hand.set_joint_pos(HAND_OPEN)

    def _hand_grasp(self) -> None:
        """Close hand to grasp position (~4.5 cm for a 5 cm cube)."""
        self._hand.set_joint_pos(HAND_GRASP)

    # ── Grasp & place ──────────────────────────────────────────────────────

    def grasp_block(self, block: BlockDetection) -> bool:
        """Execute the full grasp sequence for a single block.

        Sequence:  open hand → approach above → descend → close hand → lift

        Z calculation (2D homography, no depth):
          - block.z = z_table + BLOCK_SIZE  (top surface of cube)
          - z_grasp = z_table + GRASP_Z_OFFSET  (centre of cube, ~2.5 cm above table)

        The OBB angle is applied directly to **joint_6** (wrist rotation)
        after IK, so position accuracy is unaffected.

        Returns True if grasp succeeded.
        """
        # Use local table Z (interpolated from calibration points)
        z_tbl = self._get_table_z(block.x, block.y)
        # Gravity-sag compensation is applied inside the motion primitives now
        # (_interpolate_and_move / _refine_and_move), so target Z stays nominal.
        z_grasp = z_tbl + GRASP_Z_OFFSET               # centre of cube
        z_approach = block.z + APPROACH_HEIGHT_OFFSET  # above top surface

        # OBB angle → joint_6 offset.  Cube has 90° symmetry, so pick
        # the equivalent angle in [-45°, 45°] (smallest rotation).
        raw_deg = block.grasp_angle_deg + GRASP_YAW_OFFSET_DEG
        j6_deg = ((raw_deg + 45) % 90) - 45   # wrap to [-45°, 45°]
        j6_offset = math.radians(j6_deg)
        gx = block.x + GRASP_OFFSET_X
        gy = block.y + GRASP_OFFSET_Y

        # ── ① Open hand (wait for CAN to execute) ──────────────────────
        print("[Grasp] Opening hand ...")
        self._hand_open()
        time.sleep(CATCH_DELAY_S)

        # ── ② Approach from above (Jacobian + SLSQP fallback) ───────────
        print(f"[Grasp] Approaching above {block.class_name} at "
              f"({gx:.3f}, {gy:.3f}, {z_approach:.3f})  "
              f"j6={math.degrees(j6_offset):.0f}°")
        T_approach = self._make_target_pose(gx, gy, z_approach)
        if not self._interpolate_and_move(T_approach, j6_offset_rad=j6_offset):
            print("[Grasp] SLSQP approach failed, trying multi-restart ...")
            err = self._refine_and_move(T_approach, j6_offset_rad=j6_offset)
            if err > 0.05:
                print(f"[Grasp] SLSQP approach also failed (err={err:.3f}), aborting.")
                return False
        time.sleep(CATCH_DELAY_S)

        # ── ③ Descend to grasp point ────────────────────────────────────
        print(f"[Grasp] Descending to "
              f"({gx:.3f}, {gy:.3f}, {z_grasp:.3f})")
        T_target = self._make_target_pose(gx, gy, z_grasp)
        if not self._move_to_pose(T_target, refine=True, j6_offset_rad=j6_offset):
            print("[Grasp] Descent failed, aborting.")
            return False
        time.sleep(CATCH_DELAY_S)

        # ── ④ Close hand (wait for CAN to execute) ──────────────────────
        print("[Grasp] Closing hand to grasp position ...")
        self._hand_grasp()
        time.sleep(CATCH_DELAY_S)

        # ── ⑤ Lift ──────────────────────────────────────────────────────
        print("[Grasp] Lifting ...")
        if not self._interpolate_and_move(T_approach, j6_offset_rad=j6_offset):
            print("[Grasp] Lift failed (object may still be grasped).")
        time.sleep(CATCH_DELAY_S)

        # ── ⑥ Check grasp success (roboarm-style: is gripper in range?) ──
        current_pos = self._hand.read_joint_pos()
        # Only check the 4 main fingers (thumb + index/middle/ring), skip pinky
        active = [current_pos[i] for i in [0, 2, 3, 4]]  # thumb_pitch, idx, mid, ring
        avg_pos = sum(active) / len(active)
        # Grasp OK if fingers are in the expected range:
        #   too low → still open → no object was there
        #   too high → fully closed → nothing blocking fingers
        grasp_ok = GRASP_OK_MIN <= avg_pos <= GRASP_OK_MAX
        print(f"[Grasp] {'OK' if grasp_ok else 'FAILED'} "
              f"(avg 4-finger pos={avg_pos:.2f}, "
              f"expected [{GRASP_OK_MIN:.1f}–{GRASP_OK_MAX:.1f}])")
        return grasp_ok

    def place_block(self, place_xyz: Tuple[float, float, float]) -> bool:
        """Place the grasped block at *place_xyz*.

        Sequence:  approach above → descend → open hand → lift
        """
        px, py, pz = place_xyz
        pz_approach = pz + APPROACH_HEIGHT_OFFSET

        # ── ① Approach ─────────────────────────────────────────────────
        print(f"[Place] Approaching ({px:.3f}, {py:.3f}, {pz_approach:.3f})")
        T_approach = self._make_target_pose(px, py, pz_approach)
        if not self._interpolate_and_move(T_approach):
            print("[Place] Approach failed.")
            return False
        time.sleep(CATCH_DELAY_S)

        # ── ② Descend ──────────────────────────────────────────────────
        print(f"[Place] Descending to ({px:.3f}, {py:.3f}, {pz:.3f})")
        T_place = self._make_target_pose(px, py, pz)
        if not self._move_to_pose(T_place, refine=True):
            print("[Place] Descent failed.")
            return False
        time.sleep(CATCH_DELAY_S)

        # ── ③ Open hand (wait for CAN) ──────────────────────────────────
        print("[Place] Opening hand ...")
        self._hand_open()
        time.sleep(CATCH_DELAY_S)

        # ── ④ Lift ─────────────────────────────────────────────────────
        if not self._interpolate_and_move(T_approach):
            print("[Place] Lift failed.")
            return False

        return True

    # ── Move aside ─────────────────────────────────────────────────────────

    def _move_aside(self) -> None:
        """Park the arm to the side so it doesn't block the camera view.

        Mirrors roboarm's ``default_gripper_aside_pos`` pattern.
        """
        ax, ay, az = ASIDE_POSITION
        print(f"[Aside] Parking at ({ax:.3f}, {ay:.3f}, {az:.3f})")
        try:
            T_aside = self._make_target_pose(ax, ay, az)
            self._interpolate_and_move(T_aside)
        except Exception as e:
            print(f"[Aside] Failed: {e}")

    # ── Main loop ──────────────────────────────────────────────────────────

    def run_loop(self) -> None:
        """Run the perception–action loop.

        - **SPACE** — trigger a single grasp+place cycle (manual mode)
        - **ESC / Q** — exit
        - **A** — toggle auto-grasp mode
        """
        mode_str = "AUTO" if self._auto_grasp else "MANUAL (press SPACE)"
        if STACK_ENABLED:
            mode_str += " | STACK"
            sp = STACK_POSITION
            print(f"  Stacking at ({sp[0]:.3f}, {sp[1]:.3f})")
        print("\n" + "=" * 60)
        print(f"  D1 Block Grasp — YOLO + Dexterous Hand  [{mode_str}]")
        print(f"  Table Z = {self._z_table:.3f} m  |  "
              f"Grasp Z = {self._z_table + GRASP_Z_OFFSET:.3f} m")
        print("  SPACE = grasp  |  A = toggle auto  |  ESC/Q = quit")
        print("=" * 60 + "\n")

        window = "D1 Block Grasp" if not self._headless else None
        trigger_grasp = False
        # Track status for each detected block: SKIP / OK / FAIL
        block_status: dict = {}  # (cls_name, x, y) → status string

        try:
            while True:
                t0 = time.time()

                # ── Capture (RGB only, no depth) ────────────────────────
                rgb = self._camera.snapshot(filtered=False)

                # ── Detect (2D homography, no depth) ────────────────────
                blocks = self.detect_blocks(rgb)

                # ── Auto-detect stack height ──────────────────────────
                if STACK_ENABLED:
                    spx, spy = STACK_POSITION[:2]
                    n_stacked = sum(
                        1 for b in blocks
                        if math.sqrt((b.x - spx) ** 2 + (b.y - spy) ** 2)
                        <= PLACE_DISTANCE_THRESHOLD
                    )
                    if n_stacked > self._stack_count:
                        self._stack_count = n_stacked
                        print(f"[Stack] Auto-detected {n_stacked} blocks on tower")

                # ── Act ────────────────────────────────────────────────
                if not self._busy and blocks:
                    should_grasp = self._auto_grasp or trigger_grasp
                    trigger_grasp = False

                    if should_grasp:
                        # Pick the first block NOT already at its target
                        target = None
                        target_place = None
                        for b in blocks:
                            if STACK_ENABLED:
                                px, py = STACK_POSITION[:2]
                                # Skip blocks already on the tower
                                dist_to_stack = math.sqrt(
                                    (b.x - px) ** 2 + (b.y - py) ** 2
                                )
                                if dist_to_stack <= PLACE_DISTANCE_THRESHOLD:
                                    continue
                                target = b
                                pz = self._z_table + self._stack_count * BLOCK_SIZE + GRASP_Z_OFFSET
                                target_place = (px, py, pz)
                                break
                            else:
                                px, py = PLACE_POSITIONS.get(
                                    b.class_name,
                                    [b.x, b.y, DEFAULT_PLACE_Z],
                                )[:2]
                                dist_to_place = math.sqrt(
                                    (b.x - px) ** 2 + (b.y - py) ** 2
                                )
                                if dist_to_place > PLACE_DISTANCE_THRESHOLD:
                                    target = b
                                    target_place = (px, py, self._z_table + GRASP_Z_OFFSET)
                                    break

                        if target is None:
                            print("[Loop] All blocks already placed — done!")
                            if self._auto_grasp:
                                self._auto_grasp = False
                            continue

                        self._busy = True
                        print(f"\n[Loop] Target: {target.class_name} "
                              f"({target.score:.2f}) @ "
                              f"({target.x:.3f}, {target.y:.3f}, {target.z:.3f})"
                              f"  → place ({target_place[0]:.3f}, {target_place[1]:.3f})")

                        try:
                            if self.grasp_block(target):
                                block_status[(target.class_name, target.x, target.y)] = "OK"
                                time.sleep(1.0)
                                self.place_block(target_place)
                                if STACK_ENABLED:
                                    self._stack_count += 1
                                    print(f"[Stack] {self._stack_count} blocks stacked")
                                else:
                                    # Move aside to clear camera view (not in stack mode)
                                    self._move_aside()
                            else:
                                block_status[(target.class_name, target.x, target.y)] = "FAIL"
                                print("[Loop] Grasp failed, skipping place.")
                        except Exception as e:
                            print(f"[Loop] Error during grasp/place: {e}")
                        finally:
                            self._busy = False

                # ── Visualise ──────────────────────────────────────────
                vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for b in blocks:
                    draw_box(
                        vis, b.u, b.v, b.w, b.h,
                        np.rad2deg(b.r_rad),
                        f"{b.class_name}: {b.score:.2f}",
                    )
                    # Box centre — small grey dot for reference
                    cv2.circle(vis, (int(b.u), int(b.v)), 3, (128, 128, 128), -1)
                    # Bottom-centre — red crosshair (this is the grasp target)
                    cv2.drawMarker(
                        vis, (int(b.u_bot), int(b.v_bot)), (0, 0, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20, thickness=2,
                    )

                # Status overlay
                fps = 1.0 / max(time.time() - t0, 1e-6)
                status = "BUSY" if self._busy else (
                    "AUTO" if self._auto_grasp else "MANUAL")
                cv2.putText(vis, f"FPS: {fps:.1f}  [{status}]",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2)
                cv2.putText(vis, f"Dets: {len(blocks)}",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2)

                # Show EE position
                try:
                    qh, qa = self._get_joint_state()
                    Tee = self._kin.ee_in_base(qh, qa)
                    ex, ey, ez = Tee[0, 3], Tee[1, 3], Tee[2, 3]
                    cv2.putText(vis, f"EE: ({ex:.3f}, {ey:.3f}, {ez:.3f})",
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 200, 0), 2)
                except Exception:
                    pass

                if self._headless:
                    if self._step_count % 30 == 0:
                        cv2.imwrite("/tmp/d1_block_grasp.jpg", vis)
                else:
                    cv2.imshow(window, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key in (ord('q'), ord('Q')):
                        break
                    elif key == 32:  # SPACE
                        trigger_grasp = True
                        print("[Loop] SPACE pressed — trigger grasp")
                    elif key in (ord('a'), ord('A')):
                        self._auto_grasp = not self._auto_grasp
                        mode_str = "AUTO" if self._auto_grasp else "MANUAL"
                        print(f"[Loop] Mode: {mode_str}")

                self._step_count += 1

        except KeyboardInterrupt:
            print("\n[Loop] Interrupted by user.")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """Clean shutdown: open hand, close hardware."""
        print("[Shutdown] Opening hand ...")
        try:
            self._hand.set_joint_pos(HAND_OPEN)
            time.sleep(0.3)
        except Exception:
            pass

        print("[Shutdown] Closing hardware ...")
        try:
            self._hand.open_hand()
        except Exception:
            pass
        try:
            self._hand.close_can()
        except Exception:
            pass
        try:
            self._robot.close()
        except Exception:
            pass
        try:
            self._camera.close()
        except Exception:
            pass

        if not self._headless:
            cv2.destroyAllWindows()

        print("[Shutdown] Done.")
