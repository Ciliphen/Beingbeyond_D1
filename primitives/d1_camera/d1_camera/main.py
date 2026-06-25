#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""D1 Camera primitive — capability-based driver.

Owns ``beingbeyond/primitive/camera/*``.  Wraps Intel RealSense D435i over USB.

Capability surface (mode | transport | MCP):

  beingbeyond/primitive/camera/driver          rpc        gRPC lifecycle (built-in)
  beingbeyond/primitive/camera/info            rpc        MCP  – query model, resolution
  beingbeyond/primitive/camera/rgb             topic_out  ROS2 – continuous RGB stream
  beingbeyond/primitive/camera/depth           topic_out  ROS2 – continuous depth stream
  beingbeyond/primitive/camera/snapshot        rpc        MCP  – take one RGB frame
  beingbeyond/primitive/camera/depth_snapshot  rpc        MCP  – take one depth frame
  beingbeyond/primitive/camera/rgbd            rpc        (not MCP) – synchronised RGB-D pair
  beingbeyond/primitive/camera/intrinsics      rpc        (not MCP) – camera parameters

Image data is returned as raw uint8 bytes + width/height/encoding metadata.
Consumers (e.g. VLA policy) decode according to the encoding field.
"""
from __future__ import annotations

import os
import struct
import time

import numpy as np

from robonix_api import Primitive, Ok, Err

# ── global primitive ─────────────────────────────────────────────────────
d1_camera = Primitive(id="d1_camera", namespace="beingbeyond/primitive/camera")

# ── hardware handle ──────────────────────────────────────────────────────
_camera = None  # RealSenseCamera instance
_cfg = {}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _encode_rgb(color_rgb: np.ndarray) -> bytes:
    """Encode an RGB uint8 ndarray as raw bytes."""
    return color_rgb.astype(np.uint8).tobytes()


def _encode_depth(depth_m: np.ndarray) -> bytes:
    """Encode depth float32 (metres) as raw bytes (mm, uint16)."""
    depth_mm = (np.clip(depth_m, 0, 65.535) * 1000).astype(np.uint16)
    return depth_mm.tobytes()


def _get_intrinsics_dict(intrin) -> dict:
    """Extract fx, fy, cx, cy, distortion from a RealSense intrinsics object."""
    return {
        "fx": float(intrin.fx),
        "fy": float(intrin.fy),
        "cx": float(intrin.ppx),
        "cy": float(intrin.ppy),
        "distortion": list(intrin.coeffs) if hasattr(intrin, "coeffs") else [0.0] * 5,
        "width": int(intrin.width),
        "height": int(intrin.height),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle
# ═══════════════════════════════════════════════════════════════════════════

@d1_camera.on_init
def init(cfg: dict):
    """Light validation: check RealSense is importable, parse config."""
    global _cfg
    _cfg = {
        "width": int(cfg.get("width", 640)),
        "height": int(cfg.get("height", 480)),
        "fps": int(cfg.get("fps", 30)),
        "serial": cfg.get("serial", ""),  # optional: pick specific camera by serial
    }
    print(f"[d1_camera] init ok — {_cfg['width']}x{_cfg['height']}@{_cfg['fps']}fps")
    return Ok()


@d1_camera.on_activate
def activate():
    """Open hardware: connect to RealSense camera, start streams."""
    global _camera
    from vision import RealSenseCamera

    _camera = RealSenseCamera(
        width=_cfg["width"],
        height=_cfg["height"],
        hz=_cfg["fps"],
    )
    print(f"[d1_camera] activated — model=D435i, "
          f"{_cfg['width']}x{_cfg['height']}@{_cfg['fps']}fps")
    return Ok()


@d1_camera.on_deactivate
def deactivate():
    """Release hardware: stop camera pipeline."""
    global _camera
    if _camera is not None:
        try:
            _camera.stop()
        except Exception:
            pass
        _camera = None
    print("[d1_camera] deactivated")
    return Ok()


# ═══════════════════════════════════════════════════════════════════════════
#  prm::camera.info  (MCP ✅)
# ═══════════════════════════════════════════════════════════════════════════

@d1_camera.mcp("beingbeyond/primitive/camera/info")
def info(req) -> "camera_mcp.GetCameraInfo_Response":
    """Query camera metadata: model, resolution, fps."""
    import camera_mcp

    return camera_mcp.GetCameraInfo_Response(
        model="Intel_RealSense_D435i",
        width=_cfg["width"],
        height=_cfg["height"],
        fps=_cfg["fps"],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  prm::camera.snapshot  (MCP ✅)
# ═══════════════════════════════════════════════════════════════════════════

@d1_camera.mcp("beingbeyond/primitive/camera/snapshot")
def snapshot(req) -> "camera_mcp.GetCameraImage_Response":
    """Take one RGB frame on demand (for LLM/VLM agents)."""
    import camera_mcp

    if _camera is None:
        return camera_mcp.GetCameraImage_Response(
            image_data=b"", width=0, height=0, encoding="none",
        )
    color_rgb, _ = _camera.get_aligned_frames(filtered=False)
    h, w = color_rgb.shape[:2]
    return camera_mcp.GetCameraImage_Response(
        image_data=_encode_rgb(color_rgb),
        width=w,
        height=h,
        encoding="rgb8",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  prm::camera.depth_snapshot  (MCP ✅)
# ═══════════════════════════════════════════════════════════════════════════

@d1_camera.mcp("beingbeyond/primitive/camera/depth_snapshot")
def depth_snapshot(req) -> "camera_mcp.GetCameraImage_Response":
    """Take one depth frame on demand."""
    import camera_mcp

    if _camera is None:
        return camera_mcp.GetCameraImage_Response(
            image_data=b"", width=0, height=0, encoding="none",
        )
    _, depth_m = _camera.get_aligned_frames(filtered=False)
    h, w = depth_m.shape[:2]
    return camera_mcp.GetCameraImage_Response(
        image_data=_encode_depth(depth_m),
        width=w,
        height=h,
        encoding="16UC1",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  prm::camera.rgbd  (NOT MCP — for service/skill consumers)
# ═══════════════════════════════════════════════════════════════════════════

@d1_camera.grpc("beingbeyond/primitive/camera/rgbd",
                description="Get a synchronised RGB-D image pair.")
def rgbd(req) -> "camera_mcp.GetRGBDImage_Response":
    import camera_mcp

    if _camera is None:
        return camera_mcp.GetRGBDImage_Response(
            rgb_data=b"", depth_data=b"", width=0, height=0,
            rgb_encoding="none", depth_encoding="none", depth_scale=0.0,
        )
    color_rgb, depth_m = _camera.get_aligned_frames(filtered=False)
    h, w = color_rgb.shape[:2]
    return camera_mcp.GetRGBDImage_Response(
        rgb_data=_encode_rgb(color_rgb),
        depth_data=_encode_depth(depth_m),
        width=w,
        height=h,
        rgb_encoding="rgb8",
        depth_encoding="16UC1",
        depth_scale=1000.0,  # mm
    )


# ═══════════════════════════════════════════════════════════════════════════
#  prm::camera.intrinsics  (NOT MCP)
# ═══════════════════════════════════════════════════════════════════════════

@d1_camera.grpc("beingbeyond/primitive/camera/intrinsics",
                description="Query camera intrinsics (fx, fy, cx, cy, distortion).")
def intrinsics(req) -> "camera_mcp.GetCameraIntrinsics_Response":
    import camera_mcp

    if _camera is None:
        return camera_mcp.GetCameraIntrinsics_Response(
            fx=0.0, fy=0.0, cx=0.0, cy=0.0,
            distortion=[0.0] * 5, width=0, height=0,
        )
    d = _get_intrinsics_dict(_camera.get_camera_intrinsics())
    return camera_mcp.GetCameraIntrinsics_Response(
        fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"],
        distortion=d["distortion"],
        width=d["width"], height=d["height"],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Topic-out: prm::camera.rgb  /  prm::camera.depth
#  (ROS 2 topics, declared in on_init via declare_ros2_topic, published by
#   background thread — uncomment and wire rclpy when ROS 2 is available.)
# ═══════════════════════════════════════════════════════════════════════════


# ── main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    d1_camera.run()
