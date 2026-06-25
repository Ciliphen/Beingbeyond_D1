#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""D1 Dexterous Hand primitive — capability-based driver.

Owns ``beingbeyond/primitive/hand/*``.  Wraps the D1 DexHand SDK over CAN.

Capability surface (mode | transport | MCP):

  beingbeyond/primitive/hand/driver        rpc        gRPC lifecycle (built-in)
  beingbeyond/primitive/hand/move_joint    rpc        MCP  – set ND joint positions
  beingbeyond/primitive/hand/state_joint   topic_out  ROS2 – read joint positions
  beingbeyond/primitive/hand/state_speed   topic_out  ROS2 – read joint speeds
  beingbeyond/primitive/hand/set_speed     rpc        gRPC – set speed limits
  beingbeyond/primitive/hand/set_torque    rpc        gRPC – set torque limits
  beingbeyond/primitive/hand/info          rpc        MCP  – query metadata

All joint arrays are variable-length float64 in [0, 1] (0=open, 1=closed).
Joint count and names are self-describing via the ``info`` contract.
"""
from __future__ import annotations

import os

from robonix_api import Primitive, Ok, Err

# ── global primitive ─────────────────────────────────────────────────────
d1_hand = Primitive(id="d1_hand", namespace="beingbeyond/primitive/hand")

# ── hardware handle (set in on_activate, cleared in on_deactivate) ───────
_hand = None  # DexHand instance


# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle
# ═══════════════════════════════════════════════════════════════════════════

@d1_hand.on_init
def init(cfg: dict):
    """Light validation: check CAN interface exists, parse config."""
    global _cfg
    _cfg = {
        "hand_type": cfg.get("hand_type", "right"),
        "can_iface": cfg.get("can_iface", "can0"),
        "baudrate": int(cfg.get("baudrate", 1_000_000)),
    }

    can_iface = _cfg["can_iface"]
    can_path = f"/sys/class/net/{can_iface}/operstate"
    if not os.path.exists(can_path):
        return Err(f"CAN interface {can_iface} not present")

    print(f"[d1_hand] init ok — type={_cfg['hand_type']}, "
          f"can={_cfg['can_iface']}, br={_cfg['baudrate']}")
    return Ok()


@d1_hand.on_activate
def activate():
    """Open hardware: bring up CAN, initialise DexHand, open hand."""
    global _hand
    from beingbeyond_d1_sdk.dex_hand import DexHand

    _hand = DexHand(
        hand_type=_cfg["hand_type"],
        can_iface=_cfg["can_iface"],
        baudrate=_cfg["baudrate"],
    )
    # Ensure hand starts in a known-safe (open) state
    _hand.open_hand()
    print(f"[d1_hand] activated — joints={_hand.joint_names}")
    return Ok()


@d1_hand.on_deactivate
def deactivate():
    """Release hardware: open hand for safety, then close CAN."""
    global _hand
    if _hand is not None:
        try:
            _hand.open_hand()
        except Exception:
            pass
        try:
            _hand.close_can()
        except Exception:
            pass
        _hand = None
    print("[d1_hand] deactivated")
    return Ok()


# ═══════════════════════════════════════════════════════════════════════════
#  prm::hand.move_joint  (MCP ✅)
# ═══════════════════════════════════════════════════════════════════════════

@d1_hand.mcp("beingbeyond/primitive/hand/move_joint")
def move_joint(req) -> "hand_mcp.MoveJoint_Response":
    """Set every joint to the requested normalised position (0=open, 1=closed).

    Call ``info`` first to discover the expected array length and joint order.
    """
    import hand_mcp  # codegen MCP dataclasses

    if _hand is None:
        return hand_mcp.MoveJoint_Response(ok=False, message="hand not activated")

    positions = list(req.positions)
    n = _hand.num_joints
    if len(positions) != n:
        return hand_mcp.MoveJoint_Response(
            ok=False,
            message=f"expected {n} values (joints: {_hand.joint_names}), "
                    f"got {len(positions)}",
        )
    _hand.set_joint_pos(positions)
    return hand_mcp.MoveJoint_Response(ok=True, message="")


# ═══════════════════════════════════════════════════════════════════════════
#  prm::hand.set.speed  (NOT MCP)
# ═══════════════════════════════════════════════════════════════════════════

@d1_hand.grpc("beingbeyond/primitive/hand/set_speed",
              description="Set speed limit per joint (0-1).")
def set_speed(req) -> "hand_mcp.SetSpeed_Response":
    import hand_mcp

    if _hand is None:
        return hand_mcp.SetSpeed_Response(ok=False)
    _hand.set_speed(list(req.speed))
    return hand_mcp.SetSpeed_Response(ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  prm::hand.set.torque  (NOT MCP)
# ═══════════════════════════════════════════════════════════════════════════

@d1_hand.grpc("beingbeyond/primitive/hand/set_torque",
              description="Set torque limit per joint (0-1).")
def set_torque(req) -> "hand_mcp.SetTorque_Response":
    import hand_mcp

    if _hand is None:
        return hand_mcp.SetTorque_Response(ok=False)
    _hand.set_torque(list(req.torque))
    return hand_mcp.SetTorque_Response(ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  prm::hand.info  (MCP ✅)
# ═══════════════════════════════════════════════════════════════════════════

@d1_hand.mcp("beingbeyond/primitive/hand/info")
def info(req) -> "hand_mcp.GetHandInfo_Response":
    """Query hand metadata: joint names, count, and model.

    Call this first to discover the expected array length for ``move_joint``.
    """
    import hand_mcp

    if _hand is None:
        return hand_mcp.GetHandInfo_Response(
            joint_names=[],
            model="unknown",
            num_joints=0,
        )
    return hand_mcp.GetHandInfo_Response(
        joint_names=list(_hand.joint_names),
        model=f"D1_{_cfg['hand_type']}",
        num_joints=_hand.num_joints,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Topic-out contracts (ROS 2, declared in on_init)
#   prm::hand.state_joint  — publish joint positions on a ROS topic
#   prm::hand.state_speed  — publish joint speeds on a ROS topic
# ═══════════════════════════════════════════════════════════════════════════
#
# These are declared via d1_hand.declare_ros2_topic() and published by
# a background thread (or rclpy timer) that reads from _hand periodically.
# See developer-guide.md §14.8 for the declare_ros2_topic pattern.
#
# Minimal sketch (uncomment and wire rclpy when ROS 2 is available):
#
#   import threading, time
#   @d1_hand.on_activate
#   def activate():
#       global _hand, _pub_thread, _pub_stop
#       ...
#       d1_hand.declare_ros2_topic(
#           "beingbeyond/primitive/hand/state_joint",
#           "/d1/hand/joint_state", qos="sensor_data",
#       )
#       d1_hand.declare_ros2_topic(
#           "beingbeyond/primitive/hand/state_speed",
#           "/d1/hand/joint_speed", qos="sensor_data",
#       )
#       _pub_stop = threading.Event()
#       _pub_thread = threading.Thread(target=_publish_loop, daemon=True)
#       _pub_thread.start()
#       return Ok()
#
#   def _publish_loop():
#       while not _pub_stop.is_set():
#           if _hand:
#               # publish JointPositions msg on each topic …
#               pass
#           time.sleep(0.05)  # ~20 Hz
#


# ── main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    d1_hand.run()
