#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
D1 Camera Primitive — contract-compliant wrapper.

Implements the beingbeyond/primitive/camera/* contract surface:

    prm::camera.info            rpc        query model, resolution, fps
    prm::camera.snapshot        rpc        take one RGB frame (MCP)
    prm::camera.depth_snapshot  rpc        take one depth frame (MCP)
    prm::camera.rgbd            rpc        synchronised RGB-D pair
    prm::camera.intrinsics      rpc        camera intrinsics

Image data is returned as numpy arrays.  Consumers (e.g. VLA policy)
can encode as needed for their transport.

Usage:
    from camera.d1_camera_primitive import D1CameraPrimitive

    cam = D1CameraPrimitive(width=640, height=480, fps=30)
    info = cam.info()
    print(f"Model: {info['model']}, {info['width']}x{info['height']}")

    rgb = cam.snapshot()
    depth = cam.depth_snapshot()
    rgb, depth = cam.rgbd()
    intrinsics = cam.intrinsics()
    cam.close()
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from vision import RealSenseCamera


class D1CameraPrimitive:
    """D1 Camera primitive wrapping Intel RealSense D435i.

    Follows the beingbeyond/primitive/camera/* contract family.  Every
    method maps 1:1 to a contract; the class itself is not a Robonix
    Primitive process yet — it is the hardware-level backend.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        """Open the RealSense camera.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            fps: Frame rate.
        """
        self._width = width
        self._height = height
        self._fps = fps
        self._camera = RealSenseCamera(width=width, height=height, hz=fps)

    # ── prm::camera.info ─────────────────────────────────────────────────
    def info(self) -> Dict:
        """Query camera metadata.

        Contract: ``beingbeyond/primitive/camera/info`` (rpc, MCP).
        """
        return {
            "model": "Intel_RealSense_D435i",
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
        }

    # ── prm::camera.snapshot ─────────────────────────────────────────────
    def snapshot(self, filtered: bool = False) -> np.ndarray:
        """Take one RGB frame on demand.

        Contract: ``beingbeyond/primitive/camera/snapshot`` (rpc, MCP).

        Args:
            filtered: Apply spatial+temporal depth filtering (irrelevant for RGB).

        Returns:
            RGB image as uint8 ndarray (H, W, 3).
        """
        rgb, _ = self._camera.get_aligned_frames(filtered=False)
        return rgb

    # ── prm::camera.depth_snapshot ────────────────────────────────────────
    def depth_snapshot(self, filtered: bool = False) -> np.ndarray:
        """Take one depth frame on demand.

        Contract: ``beingbeyond/primitive/camera/depth_snapshot`` (rpc, MCP).

        Args:
            filtered: Apply spatial+temporal depth filtering.

        Returns:
            Depth as float32 ndarray (H, W), in metres.
        """
        _, depth = self._camera.get_aligned_frames(filtered=filtered)
        return depth

    # ── prm::camera.rgbd ─────────────────────────────────────────────────
    def rgbd(self, filtered: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Get a synchronised RGB-D image pair.

        Contract: ``beingbeyond/primitive/camera/rgbd`` (rpc).

        Returns:
            (rgb, depth) tuple — rgb is uint8 (H, W, 3), depth is float32 (H, W) in metres.
        """
        return self._camera.get_aligned_frames(filtered=filtered)

    # ── prm::camera.intrinsics ───────────────────────────────────────────
    def intrinsics(self) -> Dict:
        """Query camera intrinsics.

        Contract: ``beingbeyond/primitive/camera/intrinsics`` (rpc).
        """
        intrin = self._camera.get_camera_intrinsics()
        return {
            "fx": float(intrin.fx),
            "fy": float(intrin.fy),
            "cx": float(intrin.ppx),
            "cy": float(intrin.ppy),
            "distortion": (
                list(intrin.coeffs) if hasattr(intrin, "coeffs") else [0.0] * 5
            ),
            "width": int(intrin.width),
            "height": int(intrin.height),
        }

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    # ── lifecycle ─────────────────────────────────────────────────────────
    def close(self) -> None:
        """Stop camera pipeline and release USB."""
        try:
            self._camera.stop()
        except Exception:
            pass

    def __enter__(self) -> "D1CameraPrimitive":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
