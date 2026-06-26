#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
Entry point for YOLO-guided block grasping with the D1 dexterous hand.

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1
    python block_grasp/run_block_grasp.py

See ``block_grasp/config.py`` for tunable parameters.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from block_grasp.grasp_controller import BlockGraspController


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D1 Block Grasp — YOLO OBB + Dexterous Hand"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "object_detect", "runs", "积木方块", "best.pt",
        ),
        help="Path to YOLO OBB .pt checkpoint",
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
        default=640,
    )
    parser.add_argument(
        "--cam-height",
        type=int,
        default=480,
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
    args = parser.parse_args()

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
