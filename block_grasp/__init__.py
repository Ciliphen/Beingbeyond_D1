#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
Block grasp module for the D1 dexterous hand.

Integrates YOLO OBB detection, 3D coordinate transformation,
arm IK, and dexterous hand control to implement:
    感知层 → 控制层 → 规划层
"""
from __future__ import annotations
