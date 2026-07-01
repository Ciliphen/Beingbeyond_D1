#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
Coordinate utilities: pixel ↔ 3D world conversion.

Replaces roboarm's 2D homography (``pixel2pos``) with a full 3D pipeline
that leverages RealSense depth + camera intrinsics + D1Kinematics.

Pipeline
--------
pixel (u, v) + depth
    → camera_intrinsics → (X, Y, Z) in camera frame   [pixel_to_camera_3d]
    → T_base_camera      → (x, y, z) in base frame    [camera_to_base_3d]
"""
from __future__ import annotations

from typing import Dict, Tuple, Union

import cv2
import numpy as np


def pixel_to_camera_3d(
    u: float,
    v: float,
    depth_frame: np.ndarray,
    intrinsics: Dict,
    sample_radius: int = 5,
) -> Tuple[float, float, float]:
    """Convert a pixel coordinate to a 3D point in the **camera** frame.

    Uses the pinhole model with the depth value at ``(u, v)``.
    To reduce edge-noise a small patch around the pixel is sampled and
    the **median** valid depth is used.

    Args:
        u:             Horizontal pixel coordinate (column).
        v:             Vertical pixel coordinate (row).
        depth_frame:   Depth image (float32, metres), shape (H, W).
        intrinsics:    Camera intrinsics dict with keys
                       ``fx, fy, cx, cy, width, height``.
        sample_radius: Half-size of the sampling patch (pixels).

    Returns:
        ``(X, Y, Z)`` in the camera frame (metres).  Z is the depth value.

    Raises:
        ValueError:  If the sampled depth region contains only zeros/NaNs.
    """
    H, W = depth_frame.shape
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    # Sample a patch around the centre pixel
    u0 = max(0, int(u) - sample_radius)
    u1 = min(W, int(u) + sample_radius + 1)
    v0 = max(0, int(v) - sample_radius)
    v1 = min(H, int(v) + sample_radius + 1)

    patch = depth_frame[v0:v1, u0:u1]
    valid = patch[(patch > 0) & np.isfinite(patch)]

    if len(valid) == 0:
        raise ValueError(
            f"No valid depth in patch ({u0}:{u1}, {v0}:{v1}). "
            f"Check object is within camera range."
        )

    Z = float(np.median(valid))

    # Pinhole back-projection
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return X, Y, Z


def camera_to_base_3d(
    point_cam: Tuple[float, float, float],
    T_base_camera: np.ndarray,
) -> Tuple[float, float, float]:
    """Transform a 3D point from the camera frame to the robot base frame.

    Args:
        point_cam:      ``(X, Y, Z)`` in the camera frame (metres).
        T_base_camera:  4×4 homogeneous transform from camera to base.

    Returns:
        ``(x, y, z)`` in the base frame (metres).
    """
    p_cam = np.array([*point_cam, 1.0], dtype=float)  # homogeneous
    p_base = T_base_camera @ p_cam
    return float(p_base[0]), float(p_base[1]), float(p_base[2])


def pixel_to_world_2d(
    u: float,
    v: float,
    H: np.ndarray,
) -> Tuple[float, float]:
    """Convert pixel to world XY via 2D homography (roboarm ``pixel2pos``).

    Args:
        u, v:  Pixel coordinates.
        H:     3×3 homography matrix (pixel → base-frame XY平面).

    Returns:
        ``(x, y)`` in base frame (metres).
    """
    p_pix = np.array([u, v, 1.0], dtype=float)
    w = H @ p_pix
    w /= w[2]
    return float(w[0]), float(w[1])


def obb_bottom_center(
    u: float,
    v: float,
    w: float,
    h: float,
    angle_deg: float,
    ratio: float = 0.5,
) -> Tuple[float, float]:
    """Return the estimated **bottom-face centre** pixel of an OBB.

    The OBB from YOLO encloses the visible projection of the cube
    (top + side faces).  The geometric box-centre is too high and the
    bottom edge is too low.  The true table-contact point lies between
    them.

    We compute the centroid of the bottom 3 corners (the visible lower
    portion of the projection), then linearly interpolate between the
    box centre and that centroid.

    Args:
        u, v:      OBB centre (pixels).
        w, h:      OBB width & height (pixels).
        angle_deg: OBB rotation angle (degrees, OpenCV convention).
        ratio:     0.0 = box centre | 0.5 = halfway | 1.0 = bottom-3 centroid.

    Returns:
        ``(u_bot, v_bot)`` — estimated bottom-face centre in pixels.
    """
    ratio = max(0.0, min(1.0, ratio))
    box = cv2.boxPoints(((u, v), (w, h), angle_deg))  # (4, 2)
    # Sort by v (row) ascending → bottom 3 corners (largest v)
    idx = np.argsort(box[:, 1])
    u_bot3 = float(np.mean(box[idx[1:], 0]))
    v_bot3 = float(np.mean(box[idx[1:], 1]))
    # Interpolate between box centre and bottom-3 centroid
    u_out = u + ratio * (u_bot3 - u)
    v_out = v + ratio * (v_bot3 - v)
    return u_out, v_out


def estimate_grasp_angle_deg(
    u: float,
    v: float,
    w: float,
    h: float,
    angle_deg: float,
) -> float:
    """Estimate the end-effector yaw angle for grasping based on the OBB long edge.

    Mirrors roboarm's ``Arm.gripper_angle_by_longer`` logic: the gripper
    should align *perpendicular* to the longer edge of the detected box.

    Args:
        u, v:      OBB centre (pixels).
        w, h:      OBB width & height (pixels).
        angle_deg: OBB rotation angle (degrees, OpenCV convention).

    Returns:
        Recommended EE yaw angle **in degrees** around the world Z axis,
        wrapped to [-90°, +90°].
    """
    box_points = cv2.boxPoints(((u, v), (w, h), angle_deg))

    # Identify the two vertices of the longer edge
    if np.linalg.norm(box_points[0] - box_points[1]) > np.linalg.norm(
        box_points[1] - box_points[2]
    ):
        long_edge = (
            [box_points[0], box_points[1]]
            if box_points[0][0] < box_points[1][0]
            else [box_points[1], box_points[0]]
        )
    else:
        long_edge = (
            [box_points[1], box_points[2]]
            if box_points[1][0] < box_points[2][0]
            else [box_points[2], box_points[1]]
        )

    # Angle of the long edge → gripper is perpendicular
    gripper_rad = np.pi / 2 + np.arctan2(
        long_edge[1][1] - long_edge[0][1],
        long_edge[1][0] - long_edge[0][0],
    )

    # Wrap to [-π/2, π/2]
    if gripper_rad > np.pi / 2:
        gripper_rad -= np.pi

    return float(np.rad2deg(gripper_rad))
