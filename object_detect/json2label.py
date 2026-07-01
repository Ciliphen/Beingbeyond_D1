#!/usr/bin/env python3
"""
LabelMe JSON → YOLO OBB 标签 + train/val 划分。

用法:
    python object_detect/json2label.py

输入:  dataset/raw/*.jpg + *.json (LabelMe 多边形标注)
输出:  object_detect/dataset/
         images/train/  images/val/
         labels/train/  labels/val/    (YOLO OBB 格式)
         labels_preview/               (标注可视化)
"""
import json, os, shutil, sys
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "dataset", "raw")
OUT = os.path.join(ROOT, "object_detect", "dataset")
CLASS_NAMES = ["red_cube", "blue_cube", "green_cube", "yellow_cube"]
TRAIN_RATIO = 0.8

# Shift OBB centre downward so the box targets the bottom face instead of
# the full visible projection (top + side faces).  Ratio of OBB height.
# 0.0 = no shift (full cube) | 0.25 = bottom face (recommended)
OBB_V_SHIFT_RATIO = 0.25


def labelme_to_obb(img_path, json_path, label_dir, preview_dir):
    with open(json_path) as f:
        data = json.load(f)
    shapes = data.get("shapes", [])
    w, h = data["imageWidth"], data["imageHeight"]
    scale = np.array([[w, h]], dtype=float)

    lines = []
    for s in shapes:
        name = s["label"]
        if name not in CLASS_NAMES:
            print(f"  ⚠ 未知类别 '{name}': {json_path}")
            continue
        cid = CLASS_NAMES.index(name)
        pts = np.array(s["points"], dtype=np.float32)
        rect = cv2.minAreaRect(pts)          # ((cx, cy), (bw, bh), angle)
        (cx, cy), (bw, bh), angle = rect

        # Shift centre downward → target bottom face instead of full cube
        cy = cy + OBB_V_SHIFT_RATIO * bh

        box = cv2.boxPoints(((cx, cy), (bw, bh), angle)) / scale
        flat = box.reshape(-1)
        lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in flat) + "\n")

    if not lines:
        return False

    base = os.path.splitext(os.path.basename(json_path))[0]
    with open(os.path.join(label_dir, base + ".txt"), "w") as f:
        f.writelines(lines)

    # 预览图
    img = cv2.imread(img_path)
    if img is not None:
        h, w = img.shape[:2]
        for line in lines:
            parts = line.strip().split()
            cid = int(parts[0])
            xy = np.array([float(v) for v in parts[1:]]).reshape(-1, 2)
            pts = (xy * [w, h]).astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 255), 2)
            cv2.putText(img, CLASS_NAMES[cid], tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)
        cv2.imwrite(os.path.join(preview_dir, base + ".jpg"), img)

    return True


def main():
    if not os.path.isdir(SRC):
        print(f"❌ {SRC} 不存在")
        sys.exit(1)

    imgs = sorted(f for f in os.listdir(SRC)
                  if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg"})
    paired = [f for f in imgs if os.path.exists(
        os.path.join(SRC, os.path.splitext(f)[0] + ".json"))]

    missing = len(imgs) - len(paired)
    print(f"图片: {len(imgs)}  有标注: {len(paired)}" +
          (f"  缺JSON: {missing}" if missing else ""))
    if not paired:
        print("❌ 没找到已标注的图片")
        sys.exit(1)

    tmp = os.path.join(OUT, "labels_tmp")
    prev = os.path.join(OUT, "labels_preview")
    for d in [tmp, prev]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

    ok = 0
    for f in paired:
        base = os.path.splitext(f)[0]
        if labelme_to_obb(os.path.join(SRC, f), os.path.join(SRC, base + ".json"), tmp, prev):
            ok += 1
    print(f"转换: {ok}/{len(paired)}")

    # 划分 train/val
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(paired))
    n = int(len(paired) * TRAIN_RATIO)

    for split, sel in [("train", idx[:n]), ("val", idx[n:])]:
        img_d = os.path.join(OUT, "images", split)
        lbl_d = os.path.join(OUT, "labels", split)
        for d in [img_d, lbl_d]:
            if os.path.exists(d): shutil.rmtree(d)
            os.makedirs(d)
        for i in sel:
            f = paired[i]
            base = os.path.splitext(f)[0]
            shutil.copy2(os.path.join(SRC, f), os.path.join(img_d, f))
            src_l = os.path.join(tmp, base + ".txt")
            if os.path.exists(src_l):
                shutil.move(src_l, os.path.join(lbl_d, base + ".txt"))

    shutil.rmtree(tmp)
    print(f"✅ train: {n}  val: {len(paired)-n}")
    print(f"   {OUT}/images/  +  {OUT}/labels/")
    print(f"   预览: {prev}/")


if __name__ == "__main__":
    main()
