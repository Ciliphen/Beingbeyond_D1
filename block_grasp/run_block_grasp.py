#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
Entry point for YOLO-guided block grasping with the D1 dexterous hand.

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1
    python block_grasp/run_block_grasp.py                # manual mode (SPACE to trigger)
    python block_grasp/run_block_grasp.py --auto         # auto-grasp mode
    python block_grasp/run_block_grasp.py --headless     # no display window

See ``block_grasp/config.py`` for tunable parameters.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from block_grasp.grasp_controller import BlockGraspController

# Default to the latest trained model
_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "object_detect", "runs",
)

def _find_best_model() -> str:
    """Find the latest best.pt in object_detect/runs/train*/weights/."""
    runs_dir = _MODEL_DIR
    if not os.path.isdir(runs_dir):
        return os.path.join(runs_dir, "train", "weights", "best.pt")

    candidates = []
    for name in os.listdir(runs_dir):
        weights = os.path.join(runs_dir, name, "weights", "best.pt")
        if os.path.isfile(weights):
            candidates.append((os.path.getmtime(weights), weights))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    # Fallback
    return os.path.join(runs_dir, "train", "weights", "best.pt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D1 Block Grasp — YOLO OBB + Dexterous Hand"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=_find_best_model(),
        help="Path to YOLO OBB .pt checkpoint (auto-detected if omitted)",
    )
    parser.add_argument(
        "--hand-type",
        type=str,
        default="right",
        choices=["right", "left"],
        help="Dexterous hand type",
    )
    parser.add_argument(
        "--hand-can",
        type=str,
        default="can0",
        help="CAN interface for the dexterous hand",
    )
    parser.add_argument(
        "--arm-dev",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial device for the head–arm chain",
    )
    parser.add_argument(
        "--arm-baud",
        type=int,
        default=1_000_000,
        help="Serial baudrate for the head–arm chain",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="",
        help="Path to robot URDF (empty = SDK default)",
    )
    parser.add_argument(
        "--cam-width",
        type=int,
        default=1280,     # MUST match calibration resolution!
    )
    parser.add_argument(
        "--cam-height",
        type=int,
        default=720,      # MUST match calibration resolution!
    )
    parser.add_argument(
        "--cam-fps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Torch device for YOLO (cpu / cuda:0 / ...)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without OpenCV display windows",
    )
    parser.add_argument(
        "--show-depth",
        dest="show_depth",
        action="store_true",
        default=False,
        help="Show the colourised depth window (hidden by default)",
    )
    parser.add_argument(
        "--auto",
        dest="auto_grasp",
        action="store_true",
        help="Auto-grasp mode: continuously detect and grasp (default: SPACE to trigger)",
    )
    args = parser.parse_args()

    print(f"[Main] Model: {args.model}")
    print(f"[Main] Mode: {'auto-grasp' if args.auto_grasp else 'manual (SPACE to trigger)'}")

    controller = BlockGraspController(
        model_path=args.model,
        hand_type=args.hand_type,
        hand_can=args.hand_can,
        arm_dev=args.arm_dev,
        arm_baud=args.arm_baud,
        urdf_path=args.urdf,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
        cam_fps=args.cam_fps,
        device=args.device,
        headless=args.headless,
        auto_grasp=args.auto_grasp,
        show_depth=args.show_depth,
    )

    try:
        controller.run_loop()
    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")
    except Exception as e:
        print(f"\n[Main] Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
