#!/usr/bin/env python3
"""
Depth-based block detector — no training needed.

Algorithm:
  1. RANSAC plane fit on depth image → find table plane
  2. Threshold above-table points → connected components → object blobs
  3. For each blob, sample RGB → HSV → classify colour

Exports:
    DepthBlockDetector  — main class
    detect_blocks       — convenience function

Colour classification uses simple HSV ranges:
    red    H∈[0,10]∪[160,180]
    blue   H∈[90,130]
    yellow H∈[15,40]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor


# ── HSV colour ranges ──────────────────────────────────────────────────────
HSV_RANGES: Dict[str, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = {
    "red":    ((0, 10),   (80, 255), (80, 255)),   # lower
    "red2":   ((160, 180),(80, 255), (80, 255)),   # upper (wraparound)
    "blue":   ((90, 130), (80, 255), (80, 255)),
    "yellow": ((20, 35),  (80, 255), (80, 255)),
}


@dataclass
class Block:
    """A detected block in image + world coordinates."""

    label: str              # e.g. "red", "blue", "yellow"
    u: float                # centre x (pixels)
    v: float                # centre y (pixels)
    w: float                # width (pixels)
    h: float                # height (pixels)
    angle_deg: float        # rotation (degrees)
    # 3D position in camera frame (optional, if intrinsics provided)
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0


class DepthBlockDetector:
    """Detect blocks on a table using depth segmentation + RGB colour."""

    def __init__(
        self,
        table_offset_m: float = 0.02,   # how far above plane = "object" (m)
        min_area_px: int = 200,         # minimum blob area (px)
        depth_scale: float = 1000.0,    # depth units → metres (RealSense=1000)
        intrinsics: Optional[Dict] = None,
    ):
        self.table_offset_m = table_offset_m
        self.min_area_px = min_area_px
        self.depth_scale = depth_scale
        self.intrinsics = intrinsics

    def detect(self, rgb: np.ndarray, depth_m: np.ndarray) -> List[Block]:
        """Run full detection pipeline.

        Args:
            rgb:      RGB image (H, W, 3), uint8.
            depth_m:  Aligned depth image (H, W), float32 in **metres**.

        Returns:
            List of detected blocks with pixel positions and colour labels.
        """
        # Note: depth and RGB may differ in size.  We run plane detection
        # on the depth-native resolution and then scale coordinates to RGB.
        dh, dw = depth_m.shape[:2]
        rh, rw = rgb.shape[:2]
        sx = rw / dw
        sy = rh / dh

        # 1. Find table plane (at depth resolution)
        table_mask = self._find_table_mask(depth_m)

        # 2. Find connected objects above table
        objects_mask = self._objects_above_table(depth_m, table_mask)

        # 3. Upsample objects_mask to RGB resolution for extraction
        if (dw, dh) != (rw, rh):
            objects_mask = cv2.resize(
                objects_mask, (rw, rh), interpolation=cv2.INTER_NEAREST,
            )

        # 4. Extract each object's bounding box and classify colour
        blocks = self._extract_blocks(rgb, depth_m, objects_mask, sx, sy)

        return blocks

    # ── Plane fitting ──────────────────────────────────────────────────

    def _find_table_mask(self, depth_m: np.ndarray) -> np.ndarray:
        """RANSAC plane fit → mask of inlier pixels (table surface)."""
        H, W = depth_m.shape

        # Subsample valid depth points for speed
        valid = (depth_m > 0.1) & (depth_m < 5.0) & np.isfinite(depth_m)
        ys, xs = np.where(valid)

        if len(xs) < 100:
            return np.zeros((H, W), dtype=np.uint8)

        # Take every Nth sample
        step = max(1, len(xs) // 2000)
        xs_sub = xs[::step].reshape(-1, 1).astype(np.float32)
        ys_sub = ys[::step].reshape(-1, 1).astype(np.float32)
        zs_sub = depth_m[ys[::step], xs[::step]].reshape(-1, 1).astype(np.float32)

        # RANSAC: z ≈ a*x + b*y + c
        X = np.hstack([xs_sub, ys_sub])
        try:
            ransac = RANSACRegressor(
                residual_threshold=0.01,  # 1cm tolerance
                max_trials=200,
                random_state=0,
            )
            ransac.fit(X, zs_sub.ravel())
            a, b = ransac.estimator_.coef_
            c = ransac.estimator_.intercept_
        except Exception:
            return np.zeros((H, W), dtype=np.uint8)

        # Classify all valid pixels as table / not-table
        X_all = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        z_pred = a * xs + b * ys + c
        residuals = np.abs(depth_m[ys, xs] - z_pred)

        mask = np.zeros((H, W), dtype=np.uint8)
        table_idx = residuals < 0.015  # 1.5cm
        mask[ys[table_idx], xs[table_idx]] = 255

        return mask

    # ── Object segmentation ────────────────────────────────────────────

    def _objects_above_table(
        self, depth_m: np.ndarray, table_mask: np.ndarray
    ) -> np.ndarray:
        """Return mask of pixels above the table surface."""
        H, W = depth_m.shape
        valid = (depth_m > 0.1) & (depth_m < 5.0) & np.isfinite(depth_m)

        # Interpolate table depth for valid pixels
        table_depth = np.zeros_like(depth_m)
        if table_mask.any():
            # For each valid pixel, use nearest table pixel depth
            ys_t, xs_t = np.where(table_mask > 0)
            table_depth[ys_t, xs_t] = depth_m[ys_t, xs_t]
            # Dilate to fill gaps
            kernel = np.ones((7, 7), np.uint8)
            table_depth_filled = cv2.dilate(table_depth, kernel)
            mask = table_depth > 0
            # Inpaint gaps
            table_depth = cv2.inpaint(
                table_depth_filled.astype(np.float32),
                (mask == 0).astype(np.uint8),
                5, cv2.INPAINT_TELEA,
            )

        # Objects = depth is closer than table by at least offset
        above = valid & (table_depth - depth_m > self.table_offset_m)

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        above = cv2.morphologyEx(above.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        above = cv2.morphologyEx(above, cv2.MORPH_CLOSE, kernel)

        return above

    # ── Block extraction + colour classification ───────────────────────

    def _extract_blocks(
        self, rgb: np.ndarray, depth_m: np.ndarray, obj_mask: np.ndarray,
        sx: float = 1.0, sy: float = 1.0,
    ) -> List[Block]:
        """Connected components → rotated rect → colour classify."""
        # Ensure obj_mask matches RGB size
        if obj_mask.shape[:2] != rgb.shape[:2]:
            obj_mask = cv2.resize(obj_mask, (rgb.shape[1], rgb.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            obj_mask, connectivity=8
        )

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        blocks: List[Block] = []

        for i in range(1, n_labels):  # skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area_px:
                continue

            # Build object mask for this label
            obj = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            (u, v), (w, h), angle = rect

            # Classify colour from HSV within the object mask
            masked_hsv = cv2.bitwise_and(hsv, hsv, mask=obj)
            label_name = self._classify_colour(masked_hsv, obj)

            # 3D position (scale RGB coords to depth resolution for lookup)
            du = int(u / sx)
            dv = int(v / sy)
            x_m = y_m = z_m = 0.0
            if self.intrinsics and 0 <= dv < depth_m.shape[0] and 0 <= du < depth_m.shape[1]:
                z_d = float(depth_m[dv, du])
                if z_d > 0:
                    z_m = z_d
                    fx = self.intrinsics["fx"]
                    fy = self.intrinsics["fy"]
                    cx = self.intrinsics["cx"]
                    cy = self.intrinsics["cy"]
                    x_m = (u - cx) * z_m / fx
                    y_m = (v - cy) * z_m / fy

            blocks.append(Block(
                label=label_name,
                u=float(u), v=float(v),
                w=float(w), h=float(h),
                angle_deg=float(angle),
                x_m=x_m, y_m=y_m, z_m=z_m,
            ))

        return blocks

    def _classify_colour(self, masked_hsv: np.ndarray, mask: np.ndarray) -> str:
        """Classify the dominant colour of a masked HSV region."""
        valid_pixels = mask > 0
        if valid_pixels.sum() == 0:
            return "unknown"

        h_vals = masked_hsv[:, :, 0][valid_pixels]
        s_vals = masked_hsv[:, :, 1][valid_pixels]
        v_vals = masked_hsv[:, :, 2][valid_pixels]

        # Filter low-saturation / low-value pixels (noise)
        good = (s_vals > 50) & (v_vals > 50)
        if good.sum() == 0:
            return "unknown"

        h_good = h_vals[good]

        scores: Dict[str, float] = {}
        for name, (h_range, s_range, v_range) in HSV_RANGES.items():
            base = name.rstrip("2")  # "red2" → "red"
            h_lo, h_hi = h_range
            in_range = (h_good >= h_lo) & (h_good <= h_hi)
            score = in_range.sum() / len(h_good)
            scores[base] = scores.get(base, 0.0) + score

        best = max(scores, key=scores.get)
        return best if scores[best] > 0.15 else "unknown"


def detect_blocks(rgb: np.ndarray, depth_m: np.ndarray, **kwargs) -> List[Block]:
    """Convenience wrapper."""
    det = DepthBlockDetector(**kwargs)
    return det.detect(rgb, depth_m)
