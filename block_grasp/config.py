#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
Grasp configuration constants.

Mirrors the grasp-related portion of roboarm's ``config.yaml``.
All units are SI (metres, radians) unless noted otherwise.
"""
from __future__ import annotations

# ── YOLO detection ────────────────────────────────────────────────────────
CONF_THRESHOLD: float = 0.85       # confidence threshold for detections
IOU_THRESHOLD: float = 0.45        # IoU threshold for NMS

# ── Grasp motion ──────────────────────────────────────────────────────────
APPROACH_HEIGHT_OFFSET: float = 0.10   # height above target to approach first (m)
CATCH_DELAY_S: float = 0.1            # pause between motion steps (s)

# ── Dexterous hand poses (6-D normalised [0, 1]; 0=open, 1=closed) ───────
# Joint order: thumb_cmc_pitch, thumb_cmc_yaw, index_mcp_pitch,
#              middle_mcp_pitch, ring_mcp_pitch, pinky_mcp_pitch
HAND_OPEN: list[float] = [0.1, 0.1, 0.1, 0.1, 0.0, 0.0]
HAND_CLOSE: list[float] = [0.7, 0.5, 0.8, 0.8, 0.8, 0.8]

# ── Place positions per class (base-frame x, y, z in metres) ──────────────
PLACE_POSITIONS: dict[str, list[float]] = {
    "red_block":    [0.25, 0.10, 0.08],
    "blue_block":   [0.25, -0.10, 0.08],
    "yellow_block": [0.25, 0.00, 0.08],
}
DEFAULT_PLACE_Z: float = 0.08        # fallback place height (m)

# ── Depth sampling ────────────────────────────────────────────────────────
DEPTH_SAMPLE_RADIUS: int = 5          # pixel radius around detection centre
                                      # for median depth estimation
