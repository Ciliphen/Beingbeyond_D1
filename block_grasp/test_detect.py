#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
Test YOLO OBB detection with D1's RealSense camera + head control.

Opens the camera, loads a YOLO model, runs detection on every frame,
and displays annotated results.

Keyboard controls:
    ← →       head yaw   (joint_7,  ±2°/press)
    ↑ ↓       head pitch (joint_8,  ±1°/press)
    h         home head  (zero yaw + pitch)
    ESC       exit

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1
    python block_grasp/test_detect.py
    python block_grasp/test_detect.py --model object_detect/runs/积木方块/best.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import List

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from camera.d1_camera_primitive import D1CameraPrimitive
from object_detect import detect_objects_in_frame, draw_box, load_model


# ── Head control constants ────────────────────────────────────────────────
HEAD_YAW_STEP_DEG = 2.0        # left/right step (degrees)
HEAD_PITCH_STEP_DEG = 1.0       # up/down step (degrees)
HEAD_YAW_LIMIT_DEG = 90.0       # ± limit
HEAD_PITCH_LIMIT_DEG = 60.0     # ± limit


class HeadController:
    """Minimal head-yaw/pitch controller via HeadArmRobot SDK.

    Only touches the first 2 DOFs (head); arm joints are read-only passthrough.
    """

    def __init__(self, dev: str = "/dev/ttyUSB0", urdf_path: str = "", baudrate: int = 1_000_000):
        if not urdf_path:
            from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
            urdf_path = get_default_urdf_path()
        from beingbeyond_d1_sdk.head_arm import HeadArmRobot
        self._robot = HeadArmRobot(urdf_path=urdf_path, dev=dev, baudrate=baudrate)
        # Read current state
        q = self._robot.get_positions()  # 8D
        self._head_yaw = q[0]
        self._head_pitch = q[1]

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self._head_yaw)

    @property
    def pitch_deg(self) -> float:
        return math.degrees(self._head_pitch)

    def step(self, dyaw_deg: float = 0.0, dpitch_deg: float = 0.0) -> None:
        """Increment head angles by small deltas (clamped)."""
        self._head_yaw += math.radians(dyaw_deg)
        self._head_pitch += math.radians(dpitch_deg)
        self._head_yaw = max(-math.radians(HEAD_YAW_LIMIT_DEG),
                             min(math.radians(HEAD_YAW_LIMIT_DEG), self._head_yaw))
        self._head_pitch = max(-math.radians(HEAD_PITCH_LIMIT_DEG),
                               min(math.radians(HEAD_PITCH_LIMIT_DEG), self._head_pitch))
        self._send()

    def home(self) -> None:
        """Zero the head."""
        self._head_yaw = 0.0
        self._head_pitch = 0.0
        self._send()

    def _send(self) -> None:
        q = self._robot.get_positions()  # keep arm unchanged
        q[0] = self._head_yaw
        q[1] = self._head_pitch
        self._robot.set_positions(q)

    def close(self) -> None:
        self._robot.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test YOLO OBB detection + head control")
    parser.add_argument(
        "--model", type=str, default="yolo11n-obb.pt",
        help="Path to YOLO .pt checkpoint",
    )
    parser.add_argument("--conf", type=float, default=0.85, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="", help="Torch device (cpu / cuda:0)")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument(
        "--arm-dev", type=str, default="/dev/ttyUSB0",
        help="Serial device for head–arm (empty = headless, skip head control)",
    )
    parser.add_argument("--arm-baud", type=int, default=1_000_000)
    parser.add_argument("--urdf", type=str, default="", help="URDF path")
    args = parser.parse_args()

    # ── Init camera ───────────────────────────────────────────────────
    print("[Init] Opening RealSense camera ...")
    cam = D1CameraPrimitive(width=args.cam_width, height=args.cam_height, fps=args.cam_fps)
    cam_info = cam.info()
    print(f"       {cam_info['model']} {cam_info['width']}x{cam_info['height']}@{cam_info['fps']}fps")

    # ── Init head ─────────────────────────────────────────────────────
    head: HeadController | None = None
    if args.arm_dev:
        print(f"[Init] Opening head on {args.arm_dev} ...")
        head = HeadController(dev=args.arm_dev, urdf_path=args.urdf, baudrate=args.arm_baud)
        print(f"       yaw={head.yaw_deg:.0f}°  pitch={head.pitch_deg:.0f}°")
    else:
        print("[Init] No --arm-dev, head control disabled.")

    # ── Load model ────────────────────────────────────────────────────
    print(f"[Init] Loading model: {args.model}")
    model = load_model(args.model, device=args.device)
    print(f"       Classes: {list(model.names.values())}")

    # ── Main loop ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ← → head yaw  |  ↑ ↓ head pitch  |  h home  |  ESC exit")
    print("=" * 60 + "\n")

    frame_count = 0
    try:
        while True:
            t0 = time.time()

            # Grab frame
            rgb = cam.snapshot(filtered=False)

            # Detect
            detections = detect_objects_in_frame(
                model, rgb, conf_thres=args.conf, iou_thres=args.iou,
            )

            # Annotate
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for (u, v, w, h, r), score, cls_id, cls_name in detections:
                draw_box(
                    vis, u, v, w, h, np.rad2deg(r),
                    f"{cls_name}: {score:.2f}",
                )
                cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)

            # ── Overlay info ──────────────────────────────────────────
            dt = time.time() - t0
            fps = 1.0 / max(dt, 1e-6)
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(vis, f"Detections: {len(detections)}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if head:
                cv2.putText(vis, f"Head yaw: {head.yaw_deg:+.0f}  pitch: {head.pitch_deg:+.0f}",
                            (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            cv2.imshow("YOLO OBB Detection + Head Control", vis)

            # ── Keyboard (use waitKeyEx for arrow-key support) ────────
            raw = cv2.waitKeyEx(1)
            key = raw & 0xFF

            if key == 27:  # ESC
                break

            if head:
                if raw == 0x250000:       # LEFT  arrow
                    head.step(dyaw_deg=+HEAD_YAW_STEP_DEG)
                elif raw == 0x270000:     # RIGHT arrow
                    head.step(dyaw_deg=-HEAD_YAW_STEP_DEG)
                elif raw == 0x260000:     # UP    arrow
                    head.step(dpitch_deg=+HEAD_PITCH_STEP_DEG)
                elif raw == 0x280000:     # DOWN  arrow
                    head.step(dpitch_deg=-HEAD_PITCH_STEP_DEG)
                elif key == ord('h'):     # HOME
                    head.home()
                    print("[Head] Homed → yaw=0° pitch=0°")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[Test] Interrupted.")
    finally:
        print(f"[Test] {frame_count} frames processed.")
        cam.close()
        if head:
            head.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
