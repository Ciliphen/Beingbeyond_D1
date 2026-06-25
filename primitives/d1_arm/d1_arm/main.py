#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""D1 Arm primitive — capability-based driver.

Owns ``beingbeyond/primitive/arm/*``.  Wraps the D1 HeadArm SDK over serial.

Capability surface (mode | transport | MCP):

  beingbeyond/primitive/arm/driver        rpc        gRPC lifecycle (built-in)
  beingbeyond/primitive/arm/move_joint    rpc        MCP  – set ND arm joint positions
  beingbeyond/primitive/arm/state_joint   topic_out  ROS2 – read arm joint positions
  beingbeyond/primitive/arm/info          rpc        MCP  – query metadata

Only arm joints (joint_1 ~ joint_6, 6D) are exposed.  Head joints are held at
current position and will get their own primitive later.
"""
from __future__ import annotations

import os

from robonix_api import Primitive, Ok, Err

# ── constants ────────────────────────────────────────────────────────────
# SDK layout (8 joints): [0] head_yaw  [1] head_pitch  [2..7] joint_1..joint_6
_ARM_START = 2
_ARM_COUNT = 6

# ── global primitive ─────────────────────────────────────────────────────
d1_arm = Primitive(id="d1_arm", namespace="beingbeyond/primitive/arm")

# ── hardware handle ──────────────────────────────────────────────────────
_arm = None
_cfg = {}


# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle
# ═══════════════════════════════════════════════════════════════════════════

@d1_arm.on_init
def init(cfg: dict):
    global _cfg
    urdf = cfg.get("urdf_path", "")
    if not urdf:
        from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
        urdf = get_default_urdf_path()
    _cfg = {
        "urdf_path": urdf,
        "dev": cfg.get("dev", "/dev/ttyUSB0"),
        "baudrate": int(cfg.get("baudrate", 1_000_000)),
    }

    dev = _cfg["dev"]
    if not os.path.exists(dev):
        print(f"[d1_arm] WARNING: device {dev} not found (may appear later)")

    print(f"[d1_arm] init ok — dev={_cfg['dev']}, br={_cfg['baudrate']}")
    return Ok()


@d1_arm.on_activate
def activate():
    global _arm
    from beingbeyond_d1_sdk.head_arm import HeadArmRobot

    _arm = HeadArmRobot(
        urdf_path=_cfg["urdf_path"],
        dev=_cfg["dev"],
        baudrate=_cfg["baudrate"],
    )
    arm_names = _arm.joint_names[_ARM_START:_ARM_START + _ARM_COUNT]
    print(f"[d1_arm] activated — arm joints={arm_names}")
    return Ok()


@d1_arm.on_deactivate
def deactivate():
    global _arm
    if _arm is not None:
        try:
            _arm.close()
        except Exception:
            pass
        _arm = None
    print("[d1_arm] deactivated")
    return Ok()


# ═══════════════════════════════════════════════════════════════════════════
#  prm::arm.move_joint  (MCP ✅)
# ═══════════════════════════════════════════════════════════════════════════

@d1_arm.mcp("beingbeyond/primitive/arm/move_joint")
def move_joint(req) -> "arm_mcp.MoveJoint_Response":
    """Set arm joints to the requested absolute position (radians, 6D).

    Head joints are held at current position.
    Call ``info`` first to discover joint count and order.
    """
    import arm_mcp

    if _arm is None:
        return arm_mcp.MoveJoint_Response(ok=False, message="arm not activated")

    positions = list(req.positions)
    if len(positions) != _ARM_COUNT:
        return arm_mcp.MoveJoint_Response(
            ok=False,
            message=f"expected {_ARM_COUNT} arm joint values, got {len(positions)}",
        )
    full = _arm.get_positions()  # 8D
    full[_ARM_START:_ARM_START + _ARM_COUNT] = positions
    _arm.set_positions(full)
    return arm_mcp.MoveJoint_Response(ok=True, message="")


# ═══════════════════════════════════════════════════════════════════════════
#  prm::arm.info  (MCP ✅)
# ═══════════════════════════════════════════════════════════════════════════

@d1_arm.mcp("beingbeyond/primitive/arm/info")
def info(req) -> "arm_mcp.GetArmInfo_Response":
    """Query arm metadata: joint names (arm only), count, and model."""
    import arm_mcp

    if _arm is None:
        return arm_mcp.GetArmInfo_Response(
            joint_names=[], model="unknown", num_joints=0,
        )
    arm_names = _arm.joint_names[_ARM_START:_ARM_START + _ARM_COUNT]
    return arm_mcp.GetArmInfo_Response(
        joint_names=list(arm_names),
        model="D1_Arm",
        num_joints=_ARM_COUNT,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Topic-out: prm::arm.state_joint  (arm-only, 6D)
# ═══════════════════════════════════════════════════════════════════════════


# ── main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    d1_arm.run()
