#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
D1 Dexterous Hand Primitive — contract-compliant wrapper.

Implements the robonix/primitive/hand/* contract surface:

    prm::hand.move.joint      rpc        set ND joint positions (ND from info)
    prm::hand.state.joint     topic_out  read ND joint positions
    prm::hand.state.speed     topic_out  read joint speeds
    prm::hand.set.speed       rpc        set speed limit per joint
    prm::hand.set.torque      rpc        set torque limit per joint
    prm::hand.info            rpc        query joint_names, count, model

All joint-position arrays are variable-length float in [0, 1] (0=open, 1=closed).
Joint order is self-describing via info().joint_names.

Usage:
    from hand.d1_hand_primitive import D1HandPrimitive

    hand = D1HandPrimitive(hand_type="right", can_iface="can0")
    info = hand.info()
    print(f"Model: {info['model']}, joints: {info['num_joints']}")

    hand.move_joint([0.5, 0.3, 0.2, 0.2, 0.0, 0.0])
    pos = hand.state_joint()
    hand.close()
"""
from __future__ import annotations

from typing import Dict, List

from beingbeyond_d1_sdk.dex_hand import DexHand


class D1HandPrimitive:
    """D1 Dexterous Hand primitive wrapping DexHand SDK.

    Follows the robonix/primitive/hand/* contract family.  Every method
    maps 1:1 to a contract; the class itself is not a Robonix Primitive
    process yet — it is the hardware-level backend that a future
    ``Primitive(id="d1_hand", namespace="robonix/primitive/hand")``
    would call from its ``@primitive.mcp`` / ``@primitive.grpc`` handlers.
    """

    # ── joint metadata (D1-specific) ──────────────────────────────────────
    JOINT_NAMES = [
        "thumb_cmc_pitch",
        "thumb_cmc_yaw",
        "index_mcp_pitch",
        "middle_mcp_pitch",
        "ring_mcp_pitch",
        "pinky_mcp_pitch",
    ]

    def __init__(
        self,
        hand_type: str = "right",
        can_iface: str = "can0",
        baudrate: int = 1_000_000,
    ) -> None:
        """Open the D1 DexHand over CAN.

        Args:
            hand_type: ``"right"`` or ``"left"``.
            can_iface: CAN interface name (e.g. ``"can0"``).
            baudrate: CAN bitrate in bps.
        """
        self._hand_type = hand_type
        self._can_iface = can_iface
        self._hand = DexHand(
            hand_type=hand_type,
            can_iface=can_iface,
            baudrate=baudrate,
        )

    # ── prm::hand.move.joint ──────────────────────────────────────────────
    def move_joint(self, positions: List[float]) -> Dict:
        """Set every joint to the requested normalised position.

        Contract: ``robonix/primitive/hand/move_joint`` (rpc, MCP).

        Args:
            positions: Normalised joint positions in [0, 1], 0=open/1=closed.
                       Length must equal ``info()["num_joints"]``.
        Returns:
            ``{"ok": True}`` on success.
        """
        self._validate_length(positions)
        self._hand.set_joint_pos(positions)
        return {"ok": True}

    # ── prm::hand.state.joint ─────────────────────────────────────────────
    def state_joint(self) -> List[float]:
        """Read the current normalised joint positions.

        Contract: ``robonix/primitive/hand/state_joint`` (topic_out).
        """
        return self._hand.read_joint_pos()

    # ── prm::hand.state.speed ─────────────────────────────────────────────
    def state_speed(self) -> List[float]:
        """Read the current normalised joint speeds.

        Contract: ``robonix/primitive/hand/state_speed`` (topic_out).
        """
        return self._hand.get_speed()

    # ── prm::hand.set.speed ───────────────────────────────────────────────
    def set_speed(self, speed: List[float]) -> Dict:
        """Set the speed limit for each joint.

        Contract: ``robonix/primitive/hand/set_speed`` (rpc, NOT MCP).

        Args:
            speed: Normalised speed in [0, 1] per joint.
        Returns:
            ``{"ok": True}`` on success.
        """
        self._validate_length(speed)
        self._hand.set_speed(speed)
        return {"ok": True}

    # ── prm::hand.set.torque ──────────────────────────────────────────────
    def set_torque(self, torque: List[float]) -> Dict:
        """Set the torque limit for each joint.

        Contract: ``robonix/primitive/hand/set_torque`` (rpc, NOT MCP).

        Args:
            torque: Normalised torque in [0, 1] per joint.
        Returns:
            ``{"ok": True}`` on success.
        """
        self._validate_length(torque)
        self._hand.set_torque(torque)
        return {"ok": True}

    # ── prm::hand.info ────────────────────────────────────────────────────
    def info(self) -> Dict:
        """Query hand metadata — joint names, count, and model.

        Contract: ``robonix/primitive/hand/info`` (rpc, MCP).

        Returns:
            ``{"joint_names": [...], "num_joints": N, "model": "D1_v1"}``.
        """
        return {
            "joint_names": list(self.JOINT_NAMES),
            "num_joints": len(self.JOINT_NAMES),
            "model": "D1_v1",
            "hand_type": self._hand_type,
            "can_iface": self._can_iface,
        }

    # ── helpers ───────────────────────────────────────────────────────────
    def _validate_length(self, values: List[float]) -> None:
        n = len(self.JOINT_NAMES)
        if len(values) != n:
            raise ValueError(
                f"Expected {n} values (joints: {self.JOINT_NAMES}), "
                f"got {len(values)}"
            )

    # ── lifecycle ─────────────────────────────────────────────────────────
    def close(self) -> None:
        """Safely open the hand and release the CAN interface."""
        try:
            self._hand.open_hand()
        except Exception:
            pass
        try:
            self._hand.close_can()
        except Exception:
            pass

    def __enter__(self) -> "D1HandPrimitive":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
