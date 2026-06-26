#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
YOLO OBB object detection — pure detection functions.

Adapted from roboarm ``object_detect/detect.py``.  All roboarm-specific
imports (config_getter, Camera, cv2_display) have been removed so this
module can be used standalone with any camera backend.

Typical usage:

    from object_detect import load_model, detect_objects_in_frame, draw_box

    model = load_model("runs/best.pt")
    detections = detect_objects_in_frame(model, frame, conf_thres=0.85)

    for (u, v, w, h, r), score, cls_id, cls_name in detections:
        draw_box(frame, u, v, w, h, np.rad2deg(r), f"{cls_name}: {score:.2f}")
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_objects_in_frame(
    model: YOLO,
    frame: np.ndarray,
    conf_thres: float = 0.8,
    iou_thres: float = 0.45,
) -> List[Tuple[Tuple[float, ...], float, int, str]]:
    """Run YOLO OBB inference on a single RGB/BGR frame.

    Args:
        model:       A pre-loaded ``ultralytics.YOLO`` model (OBB variant).
        frame:       Input image as uint8 ndarray (H, W, 3) — RGB or BGR.
        conf_thres:  Confidence threshold (0–1).
        iou_thres:   IoU threshold for NMS.

    Returns:
        List of ``((u, v, w, h, r), score, class_id, class_name)`` where

        * ``(u, v)`` — centre of the rotated box (pixels)
        * ``(w, h)`` — width & height of the rotated box (pixels)
        * ``r``      — rotation angle **in radians**
        * ``score``  — confidence
        * ``class_id``  — integer class index
        * ``class_name`` — string class label
    """
    results = model(frame, conf=conf_thres, iou=iou_thres, verbose=False)[0]

    if results.obb is None:
        return []

    detections = results.obb.xywhr.cpu().numpy()         # (N, 5)  xywhr
    scores = results.obb.conf.cpu().numpy()               # (N,)
    class_ids = results.obb.cls.cpu().numpy().astype(int) # (N,)
    class_names = [model.names[i] for i in class_ids]

    return [
        (tuple(map(float, bbox)), float(s), int(c), str(name))
        for bbox, s, c, name in zip(detections, scores, class_ids, class_names)
    ]


def draw_box(
    frame: np.ndarray,
    u: float,
    v: float,
    w: float,
    h: float,
    angle_deg: float,
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw a rotated bounding box + label on *frame* (in-place).

    Args:
        frame:      BGR image (modified in-place).
        u, v:       Centre of the rotated box (pixels).
        w, h:       Width & height (pixels).
        angle_deg:  Rotation **in degrees** (OpenCV convention).
        label:      Text label drawn above the box.
        color:      BGR colour tuple.
        thickness:  Line thickness.
    """
    box_points = cv2.boxPoints(((u, v), (w, h), angle_deg))
    box_points = np.intp(box_points)
    cv2.drawContours(frame, [box_points], 0, color, thickness)
    cv2.putText(
        frame,
        label,
        (int(u - w / 2), int(v - h / 2) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        thickness,
    )


def load_model(model_path: str, device: str = "") -> YOLO:
    """Load a YOLO model from a ``.pt`` checkpoint.

    Args:
        model_path:  Path to the checkpoint file.
        device:      Torch device string (e.g. ``"cpu"``, ``"cuda:0"``).
                     Empty string means auto-select.

    Returns:
        An ``ultralytics.YOLO`` instance ready for inference.
    """
    model = YOLO(model_path)
    if device:
        model.to(device)
    return model
