#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
D1 Arm Primitive — contract-compliant wrapper.

Implements the beingbeyond/primitive/arm/* contract surface:

    prm::arm.move.joint      rpc        set ND joint positions (radians, absolute)
    prm::arm.state.joint     topic_out  read ND joint positions
    prm::arm.info            rpc        query joint_names, count, model

Only exposes the 6 arm joints (joint_1 ~ joint_6).  Head joints (joint_7_head_yaw,
joint_8_head_pitch) are kept at their current position and are not visible through
this primitive — they will get their own primitive later.

All joint-position arrays are float64 in radians (absolute position control).

Usage:
    from arm.d1_arm_primitive import D1ArmPrimitive

    arm = D1ArmPrimitive(dev="/dev/ttyUSB0")
    info = arm.info()
    print(f"Model: {info['model']}, joints: {info['num_joints']}")

    arm.move_joint([0.0, -0.5, 0.3, -0.2, 0.8, -0.1])
    pos = arm.state_joint()
    arm.close()
"""
from __future__ import annotations

from typing import Dict, List

from beingbeyond_d1_sdk.head_arm import HeadArmRobot

# SDK joint layout (8 total):
#   [0] joint_7_head_yaw     [1] joint_8_head_pitch
#   [2] joint_1  [3] joint_2  [4] joint_3  [5] joint_4  [6] joint_5  [7] joint_6
_ARM_START = 2   # first arm joint index in SDK array
_ARM_COUNT = 6   # number of arm joints


class D1ArmPrimitive:
    """D1 Arm primitive wrapping HeadArm SDK — arm-only (6 joints).

    Head joints are passed through transparently (held at current position)
    and are NOT exposed via info / move_joint / state_joint.
    """

    def __init__(
        self,
        urdf_path: str = "",
        dev: str = "/dev/ttyUSB0",
        baudrate: int = 1_000_000,
    ) -> None:
        """Open the D1 HeadArm over serial.

        Args:
            urdf_path: Path to the robot URDF model file.  Uses SDK default when empty.
            dev: Serial device name (e.g. ``"/dev/ttyUSB0"``).
            baudrate: Serial baudrate in bps.
        """
        if not urdf_path:
            from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
            urdf_path = get_default_urdf_path()
        self._dev = dev
        self._arm = HeadArmRobot(
            urdf_path=urdf_path,
            dev=dev,
            baudrate=baudrate,
        )

    # ── prm::arm.move.joint ──────────────────────────────────────────────
    def move_joint(self, positions: List[float]) -> Dict:
        """Set arm joints to the requested absolute position (radians).

        Contract: ``beingbeyond/primitive/arm/move_joint`` (rpc, MCP).

        Args:
            positions: 6 absolute arm joint angles in radians (joint_1 ~ joint_6).
        Returns:
            ``{"ok": True}`` on success.
        """
        self._validate_length(positions)
        full = self._arm.get_positions()          # 8D
        full[_ARM_START:_ARM_START + _ARM_COUNT] = positions
        self._arm.set_positions(full)
        return {"ok": True}

    # ── prm::arm.state.joint ─────────────────────────────────────────────
    def state_joint(self) -> List[float]:
        """Read the current arm joint positions in radians (6D).

        Contract: ``beingbeyond/primitive/arm/state_joint`` (topic_out).
        """
        return self._arm.get_positions()[_ARM_START:_ARM_START + _ARM_COUNT]

    # ── prm::arm.info ────────────────────────────────────────────────────
    def info(self) -> Dict:
        """Query arm metadata — joint names (arm only), count, and model.

        Contract: ``beingbeyond/primitive/arm/info`` (rpc, MCP).
        """
        return {
            "joint_names": self.joint_names,
            "num_joints": _ARM_COUNT,
            "model": "D1_Arm",
            "dev": self._dev,
        }

    # ── helpers ───────────────────────────────────────────────────────────
    @property
    def joint_names(self) -> List[str]:
        """Arm joint names only (joint_1 ~ joint_6)."""
        return self._arm.joint_names[_ARM_START:_ARM_START + _ARM_COUNT]

    def _validate_length(self, values: List[float]) -> None:
        if len(values) != _ARM_COUNT:
            raise ValueError(
                f"Expected {_ARM_COUNT} arm joint values, got {len(values)}"
            )

    # ── lifecycle ─────────────────────────────────────────────────────────
    def close(self) -> None:
        """Release the serial connection."""
        try:
            self._arm.close()
        except Exception:
            pass

    def __enter__(self) -> "D1ArmPrimitive":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
