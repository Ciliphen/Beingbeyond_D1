#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
Test YOLO OBB detection with D1's RealSense camera + head control.
GPU inference via a subprocess in the *bb_gpu* environment (RTX 5090).

Camera + head control run in *bb_d1*; YOLO runs in a forked *bb_gpu*
subprocess for GPU acceleration.  Use --no-gpu to run YOLO locally (CPU).

Keyboard controls:
    A / D     head yaw   left / right  (±2°/press)
    W / S     head pitch up   / down    (±1°/press)
    H         home head  (zero yaw + pitch)
    ESC / Q   exit

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1

    # GPU inference (default)
    python block_grasp/test_detect.py

    # CPU inference (no separate env needed)
    python block_grasp/test_detect.py --no-gpu
"""
from __future__ import annotations

import argparse
import io
import math
import os
import pickle
import struct
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from camera.d1_camera_primitive import D1CameraPrimitive

# Inline draw_box to avoid depending on ultralytics in bb_d1 env


def _draw_box(frame, u, v, w, h, angle_deg, label, color=(0, 255, 0), thickness=2):
    box_points = cv2.boxPoints(((u, v), (w, h), angle_deg))
    box_points = np.intp(box_points)
    cv2.drawContours(frame, [box_points], 0, color, thickness)
    cv2.putText(frame, label, (int(u - w / 2), int(v - h / 2) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)


# ── Paths ──────────────────────────────────────────────────────────────────
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BB_GPU_PYTHON = os.path.expanduser("~/miniconda3/envs/bb_gpu/bin/python")
_GPU_SERVER = os.path.join(_PROJ_ROOT, "block_grasp", "yolo_gpu_server.py")

# ── Head control constants ─────────────────────────────────────────────────
HEAD_YAW_STEP_DEG = 2.0
HEAD_PITCH_STEP_DEG = 1.0
HEAD_YAW_LIMIT_DEG = 90.0
HEAD_PITCH_LIMIT_DEG = 60.0

WINDOW = "D1 Block Detect  |  A/D yaw  W/S pitch  H home  Q/ESC quit"


class HeadController:
    """Minimal head-yaw/pitch controller via HeadArmRobot SDK."""

    def __init__(self, dev: str = "/dev/ttyUSB0", urdf_path: str = "", baudrate: int = 1_000_000):
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

    def step(self, dyaw_deg: float = 0.0, dpitch_deg: float = 0.0) -> None:
        self._head_yaw += math.radians(dyaw_deg)
        self._head_pitch += math.radians(dpitch_deg)
        self._head_yaw = max(-math.radians(HEAD_YAW_LIMIT_DEG),
                             min(math.radians(HEAD_YAW_LIMIT_DEG), self._head_yaw))
        self._head_pitch = max(-math.radians(HEAD_PITCH_LIMIT_DEG),
                               min(math.radians(HEAD_PITCH_LIMIT_DEG), self._head_pitch))
        self._send()

    def home(self) -> None:
        self._head_yaw = 0.0
        self._head_pitch = 0.0
        self._send()

    def _send(self) -> None:
        q = self._robot.get_positions()
        q[0] = self._head_yaw
        q[1] = self._head_pitch
        self._robot.set_positions(q)

    def close(self) -> None:
        self._robot.close()


# ── GPU subprocess client ──────────────────────────────────────────────────

class GPUClient:
    """Launch & communicate with the GPU YOLO server in bb_gpu env."""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        cmd = [_BB_GPU_PYTHON, _GPU_SERVER, "--model", model_path, "--device", device]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Read "ready" line from stderr
        line = self._proc.stderr.readline().decode().strip()
        print(f"       {line}")

    def infer(
        self,
        frame: np.ndarray,
        conf: float,
        iou: float,
        infer_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List, float]:
        """Send a frame, receive detections + timing.  Blocking."""
        # Encode frame as JPEG bytes (compact)
        _, img_bytes = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        payload = pickle.dumps((img_bytes.tobytes(), conf, iou, infer_size),
                               protocol=pickle.HIGHEST_PROTOCOL)
        self._proc.stdin.write(struct.pack(">I", len(payload)))
        self._proc.stdin.write(payload)
        self._proc.stdin.flush()

        # Read response
        raw_len = self._proc.stdout.read(4)
        if not raw_len:
            raise EOFError("GPU server died")
        msg_len = struct.unpack(">I", raw_len)[0]
        data = self._proc.stdout.read(msg_len)
        result = pickle.loads(data)
        return result["detections"], result["dt"]

    def close(self) -> None:
        self._proc.stdin.close()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO OBB detection + head control")
    parser.add_argument("--model", type=str,
                        default=os.path.join(_PROJ_ROOT, "object_detect", "runs",
                                            "积木方块", "best.pt"))
    parser.add_argument("--conf", type=float, default=0.85)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument("--arm-dev", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--arm-baud", type=int, default=1_000_000)
    parser.add_argument("--urdf", type=str, default="")
    parser.add_argument("--no-head", action="store_true",
                        help="Disable head control")
    parser.add_argument("--gpu-device", type=str, default="cuda:0",
                        help="CUDA device for GPU inference")
    parser.add_argument("--infer-size", type=int, nargs=2, default=[320, 240],
                        help="Width height for inference (default: 320 240)")
    parser.add_argument("--detect-every", type=int, default=3,
                        help="Run detection every N frames")
    args = parser.parse_args()

    # ── Camera (bb_d1) ─────────────────────────────────────────────────
    print("[Init] Camera ...")
    cam = D1CameraPrimitive(width=args.cam_width, height=args.cam_height, fps=args.cam_fps)
    info = cam.info()
    print(f"       {info['model']} {info['width']}x{info['height']}@{info['fps']}fps")

    # ── Head (bb_d1) ───────────────────────────────────────────────────
    head: Optional[HeadController] = None
    if not args.no_head:
        print(f"[Init] Head on {args.arm_dev} ...")
        head = HeadController(dev=args.arm_dev, urdf_path=args.urdf, baudrate=args.arm_baud)
        print(f"       yaw={head.yaw_deg:+.0f}°  pitch={head.pitch_deg:+.0f}°")

    # ── Detector (GPU subprocess in bb_gpu env) ───────────────────────
    print(f"[Init] GPU detector: {args.model}")
    detector = GPUClient(args.model, device=args.gpu_device)

    infer_size = tuple(args.infer_size)

    # ── Help ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if head:
        print("  A/D yaw  W/S pitch  H home  Q/ESC quit")
    else:
        print("  Q/ESC quit")
    print("  GPU: RTX 5090 (bb_gpu)")
    print("=" * 60 + "\n")

    # ── Loop ───────────────────────────────────────────────────────────
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    frame_count = 0
    last_detections: List = []

    try:
        while True:
            t0 = time.time()

            rgb = cam.snapshot(filtered=False)

            # ── Inference (skip frames) ────────────────────────────────
            t_infer = 0.0
            if frame_count % args.detect_every == 0:
                try:
                    dets, t_infer = detector.infer(
                        rgb, args.conf, args.iou, infer_size,
                    )
                    last_detections = dets
                except (EOFError, BrokenPipeError) as e:
                    print(f"[Error] GPU server: {e}")
                    last_detections = []

            # ── Annotate ────────────────────────────────────────────────
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for (u, v, w, h, r), score, cls_id, cls_name in last_detections:
                _draw_box(vis, u, v, w, h, np.rad2deg(r),
                         f"{cls_name}: {score:.2f}")
                cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)

            # Overlay
            dt = time.time() - t0
            fps = 1.0 / max(dt, 1e-6)
            gpu_label = "GPU" if not args.no_gpu else "CPU"
            cv2.putText(vis, f"FPS: {fps:.1f}  GPU: {t_infer*1000:.0f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, f"Dets: {len(last_detections)}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if head:
                cv2.putText(vis,
                            f"Head yaw: {head.yaw_deg:+.0f}  pitch: {head.pitch_deg:+.0f}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            cv2.imshow(WINDOW, vis)

            # ── Keyboard ───────────────────────────────────────────────
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
                    head.step(dpitch_deg=+HEAD_PITCH_STEP_DEG)
                elif key in (ord('s'), ord('S')):
                    head.step(dpitch_deg=-HEAD_PITCH_STEP_DEG)
                elif key in (ord('h'), ord('H')):
                    head.home()
                    print("[Head] → 0°")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[Test] Interrupted.")
    finally:
        print(f"[Test] {frame_count} frames.")
        cam.close()
        if head:
            head.close()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
