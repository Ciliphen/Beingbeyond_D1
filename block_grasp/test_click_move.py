#!/usr/bin/env python3
"""
Click-to-point test: verify the pixel → 3D → IK pipeline.

Click anywhere on the camera view — the arm will move the end-effector
to that 3D position, pointing straight down.

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1
    python block_grasp/test_click_move.py

Controls:
    Left-click  →  move EE to that 3D point (finger down)
    Right-click →  move EE 5cm above that point (approach)
    A/D/W/S/H   →  head control (same as test_detect.py)
    Q/ESC       →  quit
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from camera.d1_camera_primitive import D1CameraPrimitive
from block_grasp.coordinate_utils import pixel_to_camera_3d, camera_to_base_3d
from beingbeyond_d1_sdk.head_arm import HeadArmRobot

if not os.environ.get("URDF"):
    from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
    URDF = get_default_urdf_path()
else:
    URDF = os.environ["URDF"]
from beingbeyond_d1_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig

# ── Head control (same as test_detect.py) ──────────────────────────────────
HEAD_YAW_STEP = 2.0
HEAD_PITCH_STEP = 1.0
HEAD_YAW_LIMIT = 90.0
HEAD_PITCH_LIMIT = 60.0


class HeadController:
    def __init__(self, robot):
        self._r = robot
        q = self._r.get_positions()
        self._yaw = q[0]
        self._pitch = q[1]

    @property
    def yaw(self): return math.degrees(self._yaw)
    @property
    def pitch(self): return math.degrees(self._pitch)

    def step(self, dyaw=0.0, dpitch=0.0):
        self._yaw += math.radians(dyaw)
        self._pitch += math.radians(dpitch)
        self._yaw = max(-math.radians(HEAD_YAW_LIMIT), min(math.radians(HEAD_YAW_LIMIT), self._yaw))
        self._pitch = max(-math.radians(HEAD_PITCH_LIMIT), min(math.radians(HEAD_PITCH_LIMIT), self._pitch))
        self._send()

    def home(self):
        self._yaw = 0.0
        self._pitch = 0.0
        self._send()

    def _send(self):
        q = self._r.get_positions()
        q[0] = self._yaw
        q[1] = self._pitch
        self._r.set_positions(q)


# ── Marker state ───────────────────────────────────────────────────────────
_click_uv: Optional[tuple[int, int]] = None
_last_click_rgb: Optional[tuple[int, int]] = None
_last_target: Optional[np.ndarray] = None  # (x, y, z) in base frame


def _on_mouse(event, x, y, flags, param):
    global _click_uv
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_uv = (x, y)  # just record position


def main():
    global _click_uv, _last_target, _last_click_rgb

    # ── Init ───────────────────────────────────────────────────────────
    print("[Init] Camera ...")
    cam = D1CameraPrimitive(width=1280, height=720, fps=30)
    intrinsics = cam.intrinsics()
    print(f"       {intrinsics['width']}x{intrinsics['height']} "
          f"fx={intrinsics['fx']:.0f} fy={intrinsics['fy']:.0f}")

    print("[Init] Robot ...")
    robot = HeadArmRobot(urdf_path=URDF, dev="/dev/ttyUSB0", baudrate=1_000_000)
    head = HeadController(robot)
    kin = D1Kinematics(D1KinematicsConfig(urdf_path=URDF))
    print(f"       yaw={head.yaw:+.0f}°  pitch={head.pitch:+.0f}°")

    # ── Window + mouse callback ────────────────────────────────────────
    WINDOW = "Click-to-Point  |  Left=go  Right=above  WASD=head  Q=quit"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.setMouseCallback(WINDOW, _on_mouse)

    print("\n" + "=" * 60)
    print("  Left click  → move EE to surface (finger down)")
    print("  G key       → move EE to last-clicked point")
    print("  M key       → move EE 5cm above last-clicked point")
    print("  A/D/W/S/H   → head control")
    print("  Q/ESC       → quit")
    print("=" * 60 + "\n")

    frames = 0
    try:
        while True:
            t0 = time.time()

            # ── Get aligned RGB + depth ────────────────────────────────
            rgb, depth_m = cam.rgbd(filtered=True)
            if frames == 0:
                print(f"[Info] RGB shape={rgb.shape}  depth shape={depth_m.shape}")
                print(f"       intrinsics: {intrinsics['width']}x{intrinsics['height']}")

            # ── Handle click ───────────────────────────────────────────
            if _click_uv is not None:
                u, v = _click_uv
                _last_click_rgb = (u, v)
                _click_uv = None
                print(f"[Click] pixel=({u}, {v})")

            # ── Annotate ────────────────────────────────────────────────
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Always show last click position (green crosshair)
            if _last_click_rgb is not None:
                cv2.drawMarker(vis, _last_click_rgb, (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 2)

            # Draw last target
            if _last_target is not None:
                # Project back to pixel for visualization
                q_full = np.asarray(robot.get_positions(), dtype=float)
                q_head, q_arm = kin.split_q(q_full)
                T_base_cam = kin.camera_in_base(q_head, q_arm)
                T_cam_base = np.linalg.inv(T_base_cam)
                p_cam = T_cam_base @ np.array([*_last_target, 1.0])
                if p_cam[2] > 0:
                    uu = int(p_cam[0] * intrinsics["fx"] / p_cam[2] + intrinsics["cx"])
                    vv = int(p_cam[1] * intrinsics["fy"] / p_cam[2] + intrinsics["cy"])
                    cv2.drawMarker(vis, (uu, vv), (0, 0, 255),
                                   cv2.MARKER_CROSS, 30, 3)
                    cv2.putText(vis, f"({_last_target[0]:.2f},{_last_target[1]:.2f},{_last_target[2]:.2f})",
                                (uu+20, vv-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Depth visualisation (small overlay in corner)
            d_vis = depth_m.copy()
            d_vis[np.isinf(d_vis)] = 0
            d_norm = cv2.normalize(d_vis, None, 0, 255, cv2.NORM_MINMAX)
            d_colored = cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_JET)
            d_colored = cv2.resize(d_colored, (320, 180))
            vis[540:540+180, 960:960+320] = d_colored  # bottom-right corner

            # Overlay
            fps = 1.0 / max(time.time() - t0, 1e-6)
            cv2.putText(vis, f"FPS: {fps:.1f}  Head: yaw={head.yaw:+.0f} pitch={head.pitch:+.0f}",
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow(WINDOW, vis)

            # ── Keys ───────────────────────────────────────────────────
            key = cv2.waitKey(5) & 0xFF
            if key == 27 or key in (ord('q'), ord('Q')):
                break
            if key in (ord('a'), ord('A')):   head.step(dyaw=+HEAD_YAW_STEP)
            elif key in (ord('d'), ord('D')): head.step(dyaw=-HEAD_YAW_STEP)
            elif key in (ord('w'), ord('W')): head.step(dpitch=-HEAD_PITCH_STEP)
            elif key in (ord('s'), ord('S')): head.step(dpitch=+HEAD_PITCH_STEP)
            elif key in (ord('h'), ord('H')): head.home(); print("[Head] 0°")
            elif key in (ord('g'), ord('G'), ord('m'), ord('M')):
                if _last_click_rgb is None:
                    print("[Move] Click first, then press G/M")
                else:
                    u_rgb, v_rgb = _last_click_rgb
                    z_offset = 0.05 if key in (ord('m'), ord('M')) else 0.02
                    mode = "above" if key in (ord('m'), ord('M')) else "touch"
                    try:
                        dh, dw = depth_m.shape[:2]
                        rh, rw = rgb.shape[:2]
                        sx = dw / rw; sy = dh / rh
                        u_d = int(u_rgb * sx); v_d = int(v_rgb * sy)
                        d_intrin = {"fx": intrinsics["fx"]*sx, "fy": intrinsics["fy"]*sy,
                                    "cx": intrinsics["cx"]*sx, "cy": intrinsics["cy"]*sy,
                                    "width": dw, "height": dh}
                        Xc, Yc, Zc = pixel_to_camera_3d(u_d, v_d, depth_m, d_intrin, sample_radius=2)
                        # Convert RealSense optical → URDF camera frame convention
                        #   optical: X→right, Y→down, Z→forward (depth)
                        #   URDF:    X→right, Y→up,    Z→backward
                        p_cam_urdf = np.array([Xc, -Yc, -Zc, 1.0], dtype=float)
                        q_full = np.asarray(robot.get_positions(), dtype=float)
                        q_head, q_arm = kin.split_q(q_full)
                        T_base_cam = kin.camera_in_base(q_head, q_arm)
                        p_base = T_base_cam @ p_cam_urdf
                        x, y, z = float(p_base[0]), float(p_base[1]), float(p_base[2])
                        print(f"        optical=({Xc:.3f},{Yc:.3f},{Zc:.3f}) urdf=({p_cam_urdf[0]:.3f},{p_cam_urdf[1]:.3f},{p_cam_urdf[2]:.3f})")
                        print(f"        → base=({x:.3f},{y:.3f},{z:.3f})")
                        z_ee = z + z_offset
                        tgt = np.array([x, y, z_ee, 0, 0, 0, 1], dtype=float)
                        q_hs, q_as, cost, it = kin.ik_ee_quatpose_with_arm_only(tgt, q_head, q_arm)
                        print(f"[{mode.upper()}] base=({x:.3f},{y:.3f},{z:.3f}) z_ee={z_ee:.3f} cost={cost:.3f} it={it}")
                        # IK already includes head angles (q_hs); don't overwrite
                        head._yaw = q_hs[0]
                        head._pitch = q_hs[1]
                        robot.set_positions(np.concatenate([q_hs, q_as]))
                        _last_target = np.array([x, y, z_ee])
                    except ValueError as e:
                        print(f"        ❌ {e}")
                    except Exception as e:
                        print(f"        ❌ {e}")

            frames += 1

    except KeyboardInterrupt:
        print("\n[Exit]")
    finally:
        print(f"[Exit] {frames} frames.")
        cam.close()
        robot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
