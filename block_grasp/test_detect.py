#!/usr/bin/env python3
"""Test YOLO OBB detection with D1's RealSense camera + head control.

Keyboard:
    A / D     head yaw   left / right  (±2°/press)
    W / S     head pitch up   / down    (±1°/press)
    H         home head  (zero yaw + pitch)
    Q / ESC   exit

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1
    python block_grasp/test_detect.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from camera.d1_camera_primitive import D1CameraPrimitive
from object_detect import detect_objects_in_frame, draw_box, load_model

# ── Head constants ─────────────────────────────────────────────────────
HEAD_YAW_STEP_DEG = 2.0
HEAD_PITCH_STEP_DEG = 1.0
HEAD_YAW_LIMIT_DEG = 90.0
HEAD_PITCH_LIMIT_DEG = 60.0

WINDOW = "D1 Block Detect  |  A/D yaw  W/S pitch  H home  Q/ESC quit"


class HeadController:
    """Minimal head-yaw/pitch controller via HeadArmRobot SDK."""

    def __init__(self, dev="/dev/ttyUSB0", urdf_path="", baudrate=1_000_000):
        if not urdf_path:
            from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
            urdf_path = get_default_urdf_path()
        from beingbeyond_d1_sdk.head_arm import HeadArmRobot
        self._robot = HeadArmRobot(urdf_path=urdf_path, dev=dev, baudrate=baudrate)
        q = self._robot.get_positions()
        self._head_yaw = q[0]
        self._head_pitch = q[1]

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self._head_yaw)

    @property
    def pitch_deg(self) -> float:
        return math.degrees(self._head_pitch)

    def step(self, dyaw_deg=0.0, dpitch_deg=0.0):
        self._head_yaw += math.radians(dyaw_deg)
        self._head_pitch += math.radians(dpitch_deg)
        self._head_yaw = max(-math.radians(HEAD_YAW_LIMIT_DEG),
                             min(math.radians(HEAD_YAW_LIMIT_DEG), self._head_yaw))
        self._head_pitch = max(-math.radians(HEAD_PITCH_LIMIT_DEG),
                               min(math.radians(HEAD_PITCH_LIMIT_DEG), self._head_pitch))
        self._send()

    def home(self):
        self._head_yaw = 0.0
        self._head_pitch = 0.0
        self._send()

    def _send(self):
        q = self._robot.get_positions()
        q[0] = self._head_yaw
        q[1] = self._head_pitch
        self._robot.set_positions(q)

    def close(self):
        self._robot.close()


def main():
    _PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="YOLO detection + head control")
    parser.add_argument("--model", type=str,
                        default=os.path.join(_PROJ, "object_detect", "runs",
                                            "积木方块", "best.pt"))
    parser.add_argument("--conf", type=float, default=0.85)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--cam-width", type=int, default=1280)
    parser.add_argument("--cam-height", type=int, default=720)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument("--arm-dev", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--arm-baud", type=int, default=1_000_000)
    parser.add_argument("--urdf", type=str, default="")
    parser.add_argument("--no-head", action="store_true")
    parser.add_argument("--infer-w", type=int, default=416, help="Inference width")
    parser.add_argument("--infer-h", type=int, default=234, help="Inference height")
    args = parser.parse_args()

    # ── Camera ───────────────────────────────────────────────────────
    print("[Init] Camera ...")
    cam = D1CameraPrimitive(width=args.cam_width, height=args.cam_height, fps=args.cam_fps)
    info = cam.info()
    print(f"       {info['model']} {info['width']}x{info['height']}@{info['fps']}fps")

    # ── Head ─────────────────────────────────────────────────────────
    head = None
    if not args.no_head:
        print(f"[Init] Head on {args.arm_dev} ...")
        head = HeadController(dev=args.arm_dev, urdf_path=args.urdf, baudrate=args.arm_baud)
        print(f"       yaw={head.yaw_deg:+.0f}°  pitch={head.pitch_deg:+.0f}°")
    else:
        print("[Init] Head disabled (--no-head).")

    # ── Model ────────────────────────────────────────────────────────
    print(f"[Init] Model: {args.model}")
    model = load_model(args.model, device="cpu")
    print(f"       Classes: {list(model.names.values())}")

    # ── Banner ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  A/D yaw ←→   W/S pitch ↑↓   H home   Q/ESC quit")
    print("=" * 60 + "\n")

    # ── Loop ─────────────────────────────────────────────────────────
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, args.cam_width, args.cam_height)
    frame_count = 0
    infer_size = (args.infer_w, args.infer_h)
    # Shared state for background inference thread
    _lock = threading.Lock()
    _latest_rgb: Optional[np.ndarray] = None
    _latest_dets: List = []
    _latest_t_infer = 0.0
    _running = True

    def _infer_worker():
        """Run YOLO in background thread, reading latest frame and writing results."""
        nonlocal _latest_dets, _latest_t_infer
        while _running:
            with _lock:
                frame = _latest_rgb
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                small = cv2.resize(frame, infer_size, interpolation=cv2.INTER_AREA)
                t1 = time.time()
                dets_small = detect_objects_in_frame(model, small, args.conf, args.iou)
                dt = time.time() - t1
                with _lock:
                    sx = frame.shape[1] / small.shape[1]
                    sy = frame.shape[0] / small.shape[0]
                    _latest_dets = [((u*sx, v*sy, w*sx, h*sy, r), s, c, n)
                                    for (u, v, w, h, r), s, c, n in dets_small]
                    _latest_t_infer = dt
            except Exception as e:
                print(f"[Infer] Error: {e}")

    worker = threading.Thread(target=_infer_worker, daemon=True)
    worker.start()

    try:
        while True:
            t0 = time.time()

            rgb = cam.snapshot(filtered=False)

            # Feed latest frame to inference thread
            with _lock:
                _latest_rgb = rgb
                dets = list(_latest_dets)
                t_infer = _latest_t_infer

            # ── Annotate ─────────────────────────────────────────────
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for (u, v, w, h, r), score, cls_id, cls_name in dets:
                draw_box(vis, u, v, w, h, np.rad2deg(r), f"{cls_name}: {score:.2f}")
                cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)

            # Overlay
            dt = time.time() - t0
            fps = 1.0 / max(dt, 1e-6)
            cv2.putText(vis, f"FPS: {fps:.1f}  infer: {t_infer*1000:.0f}ms",
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, age_color, 3)
            cv2.putText(vis, f"Dets: {len(dets)}",
                        (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            if head:
                cv2.putText(vis,
                            f"Head: yaw={head.yaw_deg:+.0f}  pitch={head.pitch_deg:+.0f}",
                            (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)

            cv2.imshow(WINDOW, vis)

            # ── Keys ─────────────────────────────────────────────────
            raw = cv2.waitKey(5)
            key = raw & 0xFF

            if key == 27 or key in (ord('q'), ord('Q')):
                break

            if head:
                if key in (ord('a'), ord('A')):
                    head.step(dyaw_deg=+HEAD_YAW_STEP_DEG)
                elif key in (ord('d'), ord('D')):
                    head.step(dyaw_deg=-HEAD_YAW_STEP_DEG)
                elif key in (ord('w'), ord('W')):
                    head.step(dpitch_deg=-HEAD_PITCH_STEP_DEG)
                elif key in (ord('s'), ord('S')):
                    head.step(dpitch_deg=+HEAD_PITCH_STEP_DEG)
                elif key in (ord('h'), ord('H')):
                    head.home()
                    print("[Head] → 0°")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[Test] Interrupted.")
    finally:
        _running = False
        worker.join(timeout=2.0)
        print(f"[Test] {frame_count} frames.")
        cam.close()
        if head:
            head.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
