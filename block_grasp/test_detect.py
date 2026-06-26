#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
Quick smoke test for YOLO OBB detection with D1's RealSense camera.

Opens the camera, loads a YOLO model, runs detection on every frame,
and displays annotated results.  Press ESC to exit.

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1
    python block_grasp/test_detect.py
    # or with a specific model:
    python block_grasp/test_detect.py --model path/to/best.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from camera.d1_camera_primitive import D1CameraPrimitive
from object_detect import detect_objects_in_frame, draw_box, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Test YOLO OBB detection")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-obb.pt",  # nano OBB — auto-downloads on first run
        help="Path to YOLO .pt checkpoint (or model name, e.g. yolo11n-obb.pt)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.85, help="Confidence threshold"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--device", type=str, default="", help="Torch device (cpu / cuda:0)"
    )
    parser.add_argument(
        "--cam-width", type=int, default=640,
    )
    parser.add_argument(
        "--cam-height", type=int, default=480,
    )
    parser.add_argument(
        "--cam-fps", type=int, default=30,
    )
    args = parser.parse_args()

    # ── Init camera ─────────────────────────────────────────────────────
    print("[Init] Opening RealSense camera ...")
    cam = D1CameraPrimitive(width=args.cam_width, height=args.cam_height, fps=args.cam_fps)
    cam_info = cam.info()
    print(f"       {cam_info['model']} {cam_info['width']}x{cam_info['height']}@{cam_info['fps']}fps")

    # ── Load model ──────────────────────────────────────────────────────
    print(f"[Init] Loading model: {args.model}")
    model = load_model(args.model, device=args.device)
    print(f"       Classes: {list(model.names.values())}")

    # ── Main loop ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  YOLO OBB Detection Test — press ESC to exit")
    print("=" * 50 + "\n")

    frame_count = 0
    try:
        while True:
            t0 = time.time()

            # Grab frame
            rgb = cam.snapshot(filtered=False)

            # Detect
            detections = detect_objects_in_frame(
                model, rgb, conf_thres=args.conf, iou_thres=args.iou
            )

            # Annotate
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for (u, v, w, h, r), score, cls_id, cls_name in detections:
                draw_box(
                    vis, u, v, w, h, np.rad2deg(r),
                    f"{cls_name}: {score:.2f}",
                )
                # Filled centre dot
                cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)

            # FPS
            dt = time.time() - t0
            fps = 1.0 / max(dt, 1e-6)
            cv2.putText(
                vis, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2,
            )

            # Detection count
            cv2.putText(
                vis, f"Detections: {len(detections)}",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2,
            )

            # Show
            cv2.imshow("YOLO OBB Detection", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[Test] Interrupted.")
    finally:
        print(f"[Test] {frame_count} frames processed.")
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
