#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
YOLO OBB object detection module — adapted from the roboarm project.

Provides:
    - detect_objects_in_frame   — run YOLO OBB inference on a single frame
    - draw_box                  — draw rotated bounding box overlay
    - load_model                — load YOLO model from .pt checkpoint
"""
from __future__ import annotations

from .detect import detect_objects_in_frame, draw_box, load_model

__all__ = ["detect_objects_in_frame", "draw_box", "load_model"]
