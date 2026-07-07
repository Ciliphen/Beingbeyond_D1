#!/usr/bin/env python3
"""
离线检测测试 — 对现有图片运行 YOLO OBB，圈出方块并标注颜色。

不使用相机，直接读取图片文件。支持单张图片或整个目录。

Keyboard (查看模式):
    N / →     下一张
    P / ←     上一张
    S         保存标注结果到 <img>_annotated.jpg
    Q / ESC   退出

Usage:
    conda activate bb_d1
    cd ~/Beingbeyond_D1

    # 单张图片
    python block_grasp/test_detect_image.py --image dataset/raw/00001.jpg

    # 整个目录（可翻页浏览）
    python block_grasp/test_detect_image.py --dir dataset/raw

    # 批量保存标注结果，不弹窗
    python block_grasp/test_detect_image.py --dir dataset/raw --save-all --no-show
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import List

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from object_detect import detect_objects_in_frame, draw_box, load_model

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW = "Detect Image  |  N/P next/prev  |  S save  |  Q quit"

# 每个类别一个颜色（BGR），方便区分
CLASS_COLORS = {
    "red_cube":    (0, 0, 255),
    "blue_cube":   (255, 0, 0),
    "green_cube":  (0, 255, 0),
    "yellow_cube": (0, 255, 255),
}


def _find_best_model() -> str:
    """自动选择 object_detect/runs 下最新的 best.pt。"""
    runs = os.path.join(_ROOT, "object_detect", "runs")
    candidates = []
    if os.path.isdir(runs):
        for name in os.listdir(runs):
            w = os.path.join(runs, name, "weights", "best.pt")
            if os.path.isfile(w):
                candidates.append((os.path.getmtime(w), w))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return os.path.join(runs, "train", "weights", "best.pt")


def annotate(img_bgr: np.ndarray, model, conf: float, iou: float,
             infer_size: tuple = None) -> tuple:
    """对一张 BGR 图片运行检测并绘制标注。返回 (标注图, 检测数)。

    infer_size: (w, h) 推理分辨率。None=原图；给定则先缩放再推理
                （复现 test_detect.py 相机版的 416×234 行为）。
    """
    t0 = time.time()
    if infer_size is not None:
        small = cv2.resize(img_bgr, infer_size, interpolation=cv2.INTER_AREA)
        dets_s = detect_objects_in_frame(model, small, conf, iou)
        # 检测坐标从缩放图映射回原图
        sx = img_bgr.shape[1] / small.shape[1]
        sy = img_bgr.shape[0] / small.shape[0]
        dets = [((u * sx, v * sy, w * sx, h * sy, r), s, c, n)
                for (u, v, w, h, r), s, c, n in dets_s]
    else:
        dets = detect_objects_in_frame(model, img_bgr, conf, iou)
    dt = time.time() - t0

    vis = img_bgr.copy()
    for (u, v, w, h, r), score, _, name in dets:
        color = CLASS_COLORS.get(name, (0, 255, 0))
        draw_box(vis, u, v, w, h, np.rad2deg(r), f"{name}: {score:.2f}", color=color)
        cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)

    cv2.putText(vis, f"Dets: {len(dets)}  infer: {dt*1000:.0f}ms",
                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return vis, len(dets)


def main():
    p = argparse.ArgumentParser(description="离线图片 YOLO OBB 检测")
    p.add_argument("--image", default="", help="单张图片路径")
    p.add_argument("--dir", default="", help="图片目录（浏览所有图片）")
    p.add_argument("--model", default="", help="模型路径（默认自动选最新）")
    p.add_argument("--conf", type=float, default=0.85)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--infer-size", default="",
                   help="推理分辨率 WxH，如 416x234 复现相机版；留空用原图")
    p.add_argument("--save-all", action="store_true", help="批量保存标注结果")
    p.add_argument("--no-show", action="store_true", help="不弹窗显示")
    args = p.parse_args()

    # 解析推理分辨率
    infer_size = None
    if args.infer_size:
        w, h = args.infer_size.lower().split("x")
        infer_size = (int(w), int(h))
        print(f"[Infer] 推理分辨率 {infer_size[0]}x{infer_size[1]} (复现相机版)")

    # ── 收集图片列表 ─────────────────────────────────────────────────
    if args.image:
        paths = [args.image]
    elif args.dir:
        paths = sorted(
            glob.glob(os.path.join(args.dir, "*.jpg"))
            + glob.glob(os.path.join(args.dir, "*.png"))
        )
    else:
        print("请用 --image <文件> 或 --dir <目录> 指定输入")
        return

    if not paths:
        print("未找到任何图片")
        return
    print(f"[Input] {len(paths)} 张图片")

    # ── 加载模型 ─────────────────────────────────────────────────────
    model_path = args.model or _find_best_model()
    print(f"[Model] {model_path}")
    model = load_model(model_path, device="cpu")
    print(f"[Model] Classes: {list(model.names.values())}")

    # ── 批量保存模式 ─────────────────────────────────────────────────
    if args.save_all:
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                print(f"  ⚠ 无法读取 {path}")
                continue
            vis, n = annotate(img, model, args.conf, args.iou, infer_size)
            base, ext = os.path.splitext(path)
            out = f"{base}_annotated{ext}"
            cv2.imwrite(out, vis)
            print(f"  {os.path.basename(path)}: {n} 个检测 → {os.path.basename(out)}")
        if args.no_show:
            return

    # ── 浏览模式 ─────────────────────────────────────────────────────
    if args.no_show:
        return

    print("\n" + "=" * 60)
    print("  N/→ 下一张  |  P/← 上一张  |  S 保存  |  Q/ESC 退出")
    print("=" * 60 + "\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    idx = 0
    cache: dict = {}   # path → (vis, n)

    while True:
        path = paths[idx]
        if path not in cache:
            img = cv2.imread(path)
            if img is None:
                print(f"⚠ 无法读取 {path}")
                idx = (idx + 1) % len(paths)
                continue
            cache[path] = annotate(img, model, args.conf, args.iou, infer_size)
        vis, n = cache[path]

        # 标题栏叠加当前文件信息
        disp = vis.copy()
        cv2.putText(disp, f"[{idx+1}/{len(paths)}] {os.path.basename(path)}",
                    (15, disp.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 200, 0), 2)
        cv2.imshow(WINDOW, disp)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('n'), ord('N'), 83, 3):   # N or → (83/3 arrow codes vary)
            idx = (idx + 1) % len(paths)
        elif key in (ord('p'), ord('P'), 81, 2):   # P or ←
            idx = (idx - 1) % len(paths)
        elif key in (ord('s'), ord('S')):
            base, ext = os.path.splitext(path)
            out = f"{base}_annotated{ext}"
            cv2.imwrite(out, vis)
            print(f"  已保存 → {out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
