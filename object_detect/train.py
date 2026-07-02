#!/usr/bin/env python3
"""
YOLO OBB 训练脚本

用法:
    conda activate bb_gpu
    cd ~/Beingbeyond_D1
    python object_detect/train.py
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    model = YOLO("yolo11x-obb.pt")  # 首次运行会自动下载（约56MB）

    model.train(
        data=os.path.join(ROOT, "data.yaml"),
        imgsz=640,
        epochs=1000,
        batch=8,
        workers=4,
        close_mosaic=10,
        device="0",
        amp=False,
        optimizer="SGD",
        plots=False,       # skip PR-curve plotting (matplotlib font bug)
        project=os.path.join(ROOT, "runs"),
        name="train",
    )
