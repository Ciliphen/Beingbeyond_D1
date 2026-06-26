#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
Block grasp controller for the D1 dexterous hand.

Integrates the three-layer architecture:

    Perception  →  YOLO OBB detection + RealSense camera
    Control     →  D1ArmPrimitive (IK) + D1HandPrimitive (fingers)
    Planning    →  detect → transform → plan → execute grasp/place

Mirrors roboarm's ``chess/catch_and_place.py`` + ``arm/arm_base.py`` but
adapted for the D1's 3D perception pipeline and 6-finger dexterous hand.
"""
from __future__ import annotations

import concurrent.futures
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

from arm.d1_arm_primitive import D1ArmPrimitive
from camera.d1_camera_primitive import D1CameraPrimitive
from hand.d1_hand_primitive import D1HandPrimitive
from object_detect.detect import (
    detect_objects_in_frame,
    draw_box,
    load_model,
)
from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig

from .config import (
    APPROACH_HEIGHT_OFFSET,
    CATCH_DELAY_S,
    CONF_THRESHOLD,
    DEFAULT_PLACE_Z,
    DEPTH_SAMPLE_RADIUS,
    HAND_CLOSE,
    HAND_OPEN,
    IOU_THRESHOLD,
    PLACE_POSITIONS,
)
from .coordinate_utils import (
    camera_to_base_3d,
    estimate_grasp_angle_deg,
    pixel_to_camera_3d,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BlockDetection:
    """A single detected block with its 3D position resolved."""

    class_name: str
    class_id: int
    score: float
    # Pixel-space OBB
    u: float
    v: float
    w: float
    h: float
    r_rad: float
    # World-space (base frame)
    x: float
    y: float
    z: float
    # Recommended EE yaw (degrees)
    grasp_angle_deg: float


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class BlockGraspController:
    """Top-level controller for YOLO-guided block grasping with D1.

    Usage::

        ctrl = BlockGraspController(
            model_path="object_detect/runs/best.pt",
            hand_type="right", hand_can="can0",
            arm_dev="/dev/ttyUSB0",
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
        """
        self._headless = headless

        # ── Perception ─────────────────────────────────────────────────
        print("[Init] Opening RealSense camera ...")
        self._camera = D1CameraPrimitive(
            width=cam_width, height=cam_height, fps=cam_fps
        )
        self._intrinsics = self._camera.intrinsics()

        print(f"[Init] Loading YOLO model from {model_path} ...")
        self._model = load_model(model_path, device=device)

        # ── Control ────────────────────────────────────────────────────
        # URDF path
        if not urdf_path:
            from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
            urdf_path = get_default_urdf_path()

        print(f"[Init] Opening head–arm on {arm_dev} ...")
        self._arm = D1ArmPrimitive(dev=arm_dev, urdf_path=urdf_path, baudrate=arm_baud)

        print(f"[Init] Opening dexterous hand on {hand_can} ({hand_type}) ...")
        self._hand = D1HandPrimitive(hand_type=hand_type, can_iface=hand_can)

        # Kinematics
        print("[Init] Setting up kinematics ...")
        self._kin = D1Kinematics(D1KinematicsConfig(urdf_path=urdf_path))

        # ── Planning state ─────────────────────────────────────────────
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._future: Optional[concurrent.futures.Future] = None
        self._step_count = 0

        print("[Init] Ready.")

    # ── Perception helpers ─────────────────────────────────────────────────

    def detect_blocks(
        self,
        frame: np.ndarray,
        depth_frame: np.ndarray,
    ) -> List[BlockDetection]:
        """Detect blocks in a colour frame and resolve them to 3D.

        Args:
            frame:       RGB colour image (H, W, 3).
            depth_frame: Depth image in metres (H, W).

        Returns:
            List of resolved ``BlockDetection``, sorted by score descending.
        """
        detections = detect_objects_in_frame(
            self._model, frame, conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD
        )

        # Read current joint state for camera-in-base transform
        q_full = self._arm._arm.get_positions()  # 8-D
        q_head, q_arm = self._kin.split_q(np.asarray(q_full, dtype=float))
        T_base_cam = self._kin.camera_in_base(q_head, q_arm)

        blocks: List[BlockDetection] = []
        for (u, v, w, h, r), score, cls_id, cls_name in detections:
            try:
                # Pixel → camera 3D
                Xc, Yc, Zc = pixel_to_camera_3d(
                    u, v, depth_frame, self._intrinsics,
                    sample_radius=DEPTH_SAMPLE_RADIUS,
                )
                # Camera → base 3D
                x, y, z = camera_to_base_3d((Xc, Yc, Zc), T_base_cam)

                grasp_angle_deg = estimate_grasp_angle_deg(
                    u, v, w, h, np.rad2deg(r)
                )

                blocks.append(BlockDetection(
                    class_name=cls_name,
                    class_id=cls_id,
                    score=score,
                    u=u, v=v, w=w, h=h, r_rad=r,
                    x=x, y=y, z=z,
                    grasp_angle_deg=grasp_angle_deg,
                ))
            except ValueError as e:
                print(f"[Detect] Skipping {cls_name} at ({u:.0f},{v:.0f}): {e}")
                continue

        blocks.sort(key=lambda b: b.score, reverse=True)
        return blocks

    # ── Motion primitives ──────────────────────────────────────────────────

    def _hand_open(self) -> None:
        """Open the dexterous hand."""
        self._hand.move_joint(HAND_OPEN)

    def _hand_close(self) -> None:
        """Close the dexterous hand for a power grasp."""
        self._hand.move_joint(HAND_CLOSE)

    def _compute_topdown_quatpose(
        self,
        x: float,
        y: float,
        z: float,
        yaw_deg: float,
    ) -> np.ndarray:
        """Build a 7-D quatpose (x, y, z, qx, qy, qz, qw) for a top-down grasp.

        EE Z points downward (world -Z), EE X aligns with *yaw_deg* in the
        world XY plane.
        """
        yaw = math.radians(yaw_deg)
        R_ee = R.from_matrix([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw),  math.cos(yaw), 0],
            [0,              0,            -1],
        ])
        qx, qy, qz, qw = R_ee.as_quat()  # SciPy: xyzw
        return np.array([x, y, z, qx, qy, qz, qw], dtype=float)

    def _ik_solve_and_move(
        self,
        quatpose: np.ndarray,
        timeout_s: float = 10.0,
    ) -> bool:
        """Run arm-only IK and command the robot.  Returns True on success."""
        q_full = np.asarray(self._arm._arm.get_positions(), dtype=float)
        q_head, q_arm = self._kin.split_q(q_full)

        try:
            q_head_sol, q_arm_sol, cost, inner_iters = (
                self._kin.ik_ee_quatpose_with_arm_only(quatpose, q_head, q_arm)
            )
        except Exception as e:
            print(f"[IK] Solver failed: {e}")
            return False

        print(
            f"[IK] cost={cost:.4f}, inner_iters={inner_iters}, "
            f"arm_deg={[round(math.degrees(v),1) for v in q_arm_sol]}"
        )

        q_cmd = np.concatenate([q_head_sol, q_arm_sol])
        self._arm._arm.set_positions(q_cmd)

        try:
            dt = self._arm._arm.wait_until_reached(
                q_cmd,
                active_joint_indices=range(2, 8),  # arm joints only
                pos_tol_deg=5.0,
            )
            if dt is None:
                print("[IK] Timed out waiting for convergence.")
                return False
        except Exception as e:
            print(f"[IK] wait_until_reached error: {e}")
            return False

        return True

    def grasp_block(self, block: BlockDetection) -> bool:
        """Execute the full grasp sequence for a single block.

        Sequence:  approach above → descend → close hand → lift

        Returns True if grasp succeeded (hand not fully open after close).
        """
        z_target = block.z
        z_approach = z_target + APPROACH_HEIGHT_OFFSET

        # ── ① Approach from above ──────────────────────────────────────
        print(f"[Grasp] Approaching above {block.class_name} at "
              f"({block.x:.3f}, {block.y:.3f}, {z_approach:.3f})")
        self._hand_open()
        time.sleep(CATCH_DELAY_S)

        qp_approach = self._compute_topdown_quatpose(
            block.x, block.y, z_approach, block.grasp_angle_deg
        )
        if not self._ik_solve_and_move(qp_approach):
            print("[Grasp] Approach failed, aborting.")
            return False
        time.sleep(CATCH_DELAY_S * 2)

        # ── ② Descend to grasp point ───────────────────────────────────
        print(f"[Grasp] Descending to ({block.x:.3f}, {block.y:.3f}, {z_target:.3f})")
        qp_target = self._compute_topdown_quatpose(
            block.x, block.y, z_target, block.grasp_angle_deg
        )
        if not self._ik_solve_and_move(qp_target):
            print("[Grasp] Descent failed, aborting.")
            return False
        time.sleep(CATCH_DELAY_S)

        # ── ③ Close hand ───────────────────────────────────────────────
        print("[Grasp] Closing hand ...")
        self._hand_close()
        time.sleep(CATCH_DELAY_S)

        # ── ④ Lift ─────────────────────────────────────────────────────
        print("[Grasp] Lifting ...")
        if not self._ik_solve_and_move(qp_approach):
            print("[Grasp] Lift failed (object may still be grasped).")
            # Don't return False — we may still have the block
        time.sleep(CATCH_DELAY_S)

        # ── ⑤ Check grasp success ──────────────────────────────────────
        # For dexterous hand: check if fingers are near closed position
        # (not fully open = something is in the hand)
        current_pos = self._hand.state_joint()
        avg_pos = sum(current_pos) / len(current_pos)
        grasp_ok = avg_pos > 0.3  # heuristic
        print(f"[Grasp] {'OK' if grasp_ok else 'FAILED'} (avg finger pos={avg_pos:.2f})")
        return grasp_ok

    def place_block(self, place_xyz: Tuple[float, float, float]) -> bool:
        """Place the grasped block at *place_xyz*.

        Sequence:  approach above → descend → open hand → lift
        """
        px, py, pz = place_xyz
        pz_approach = pz + APPROACH_HEIGHT_OFFSET

        # ── ① Approach ─────────────────────────────────────────────────
        print(f"[Place] Approaching ({px:.3f}, {py:.3f}, {pz_approach:.3f})")
        qp_approach = self._compute_topdown_quatpose(px, py, pz_approach, 0.0)
        if not self._ik_solve_and_move(qp_approach):
            print("[Place] Approach failed.")
            return False
        time.sleep(CATCH_DELAY_S * 2)

        # ── ② Descend ──────────────────────────────────────────────────
        print(f"[Place] Descending to ({px:.3f}, {py:.3f}, {pz:.3f})")
        qp_place = self._compute_topdown_quatpose(px, py, pz, 0.0)
        if not self._ik_solve_and_move(qp_place):
            print("[Place] Descent failed.")
            return False
        time.sleep(CATCH_DELAY_S)

        # ── ③ Open hand ────────────────────────────────────────────────
        print("[Place] Opening hand ...")
        self._hand_open()
        time.sleep(CATCH_DELAY_S)

        # ── ④ Lift ─────────────────────────────────────────────────────
        if not self._ik_solve_and_move(qp_approach):
            print("[Place] Lift failed.")
            return False

        return True

    # ── Main loop ──────────────────────────────────────────────────────────

    def run_loop(self) -> None:
        """Run the perception–action loop indefinitely.

        Press **Esc** in the display window to exit.
        """
        print("\n" + "=" * 50)
        print("  D1 Block Grasp — YOLO + Dexterous Hand")
        print("  Press ESC to exit")
        print("=" * 50 + "\n")

        window = "D1 Block Grasp" if not self._headless else None

        try:
            while True:
                t0 = time.time()

                # ── Capture ────────────────────────────────────────────
                rgb, depth = self._camera.rgbd(filtered=False)

                # ── Detect ─────────────────────────────────────────────
                blocks = self.detect_blocks(rgb, depth)

                # ── Act (async, non-blocking) ──────────────────────────
                idle = self._future is None or self._future.done()

                if blocks and idle:
                    target = blocks[0]
                    print(f"\n[Loop] Target: {target.class_name} "
                          f"({target.score:.2f}) @ "
                          f"({target.x:.3f}, {target.y:.3f}, {target.z:.3f})")

                    place_name = target.class_name
                    place_xyz = PLACE_POSITIONS.get(
                        place_name,
                        [target.x, target.y, DEFAULT_PLACE_Z],
                    )

                    def _do_grasp_and_place():
                        if self.grasp_block(target):
                            time.sleep(1.0)
                            self.place_block(tuple(place_xyz))
                        else:
                            print("[Loop] Grasp failed, skipping place.")

                    self._future = self._executor.submit(_do_grasp_and_place)

                # ── Visualise ───────────────────────────────────────────
                vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for b in blocks:
                    draw_box(
                        vis, b.u, b.v, b.w, b.h,
                        np.rad2deg(b.r_rad),
                        f"{b.class_name}: {b.score:.2f}",
                    )
                    # Draw grasp point crosshair
                    cx, cy = int(b.u), int(b.v)
                    cv2.drawMarker(
                        vis, (cx, cy), (0, 0, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20, thickness=2,
                    )

                fps = 1.0 / max(time.time() - t0, 1e-6)
                cv2.putText(
                    vis, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2,
                )

                if self._headless:
                    # Save periodically for headless debugging
                    if self._step_count % 30 == 0:
                        cv2.imwrite("/tmp/d1_block_grasp.jpg", vis)
                else:
                    cv2.imshow(window, vis)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break

                self._step_count += 1

        except KeyboardInterrupt:
            print("\n[Loop] Interrupted by user.")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """Clean shutdown: stop executor, open hand, close hardware."""
        print("[Shutdown] Stopping executor ...")
        self._executor.shutdown(wait=True, cancel_futures=True)

        print("[Shutdown] Opening hand ...")
        try:
            self._hand.move_joint(HAND_OPEN)
            time.sleep(0.3)
        except Exception:
            pass

        print("[Shutdown] Closing hardware ...")
        try:
            self._hand.close()
        except Exception:
            pass
        try:
            self._arm.close()
        except Exception:
            pass
        try:
            self._camera.close()
        except Exception:
            pass

        if not self._headless:
            cv2.destroyAllWindows()

        print("[Shutdown] Done.")
