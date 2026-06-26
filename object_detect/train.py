#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
YOLOv11-OBB training script — adapted from roboarm ``object_detect/train.py``.

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1
    python object_detect/train.py
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11x-obb.pt")

    model.train(
        data=os.path.join(os.path.dirname(__file__), "data.yaml"),
        cache=False,
        imgsz=640,
        epochs=1000,
        batch=32,
        close_mosaic=10,
        device="0",
        optimizer="SGD",
        project=os.path.join(os.path.dirname(__file__), "runs", "train"),
        name="exp",
    )
