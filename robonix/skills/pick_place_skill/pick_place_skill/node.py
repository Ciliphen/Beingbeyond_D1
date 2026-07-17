# SPDX-License-Identifier: MulanPSL-2.0
"""Pick-Place Skill Node.

Exposes a coordinate-based pick-and-place as Robonix Skill MCP tools, so Pilot
can trigger it with natural language via ``rbnx chat``.

Tools:
  - pick_and_place(pick, offset)   — grasp at absolute ``pick``, release at ``pick + offset``
  - move_home()                    — open hand + park arm clear of the workspace

Follows the block_grasp_skill pattern: a raw FastMCP app attached to the Skill,
capabilities declared manually after bootstrap. On first tool call the skill
connects the d1 arm/hand *primitives* over gRPC (the primitives own the
hardware) and drives the pick/place through them — it never opens the serial
link / CAN bus itself, and (unlike block_grasp) uses no camera and no vision.
Kinematics/IK stay local (pure compute).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback

# -- Make the Beingbeyond_D1 repo importable (block_grasp kinematics/ik/config) --
_ROOT = os.environ.get("BEINGBEYOND_PATH", "$HOME/Beingbeyond_D1")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mcp.server.fastmcp import FastMCP
from robonix_api import Skill

# ---------------------------------------------------------------------------
# Skill provider + MCP app
# ---------------------------------------------------------------------------

_NAMESPACE = "robonix/skill/pick_place"

skill = Skill(id="pick_place", namespace=_NAMESPACE)
mcp = FastMCP("pick-place")

# Lazy-initialised controller (built on first tool call → fast bootstrap).
_controller = None
_init_lock = threading.Lock()


def _ensure_controller():
    """Late-init the controller on first use. Discovers and connects the d1
    arm/hand *primitives* via atlas (the primitives own the hardware; the skill
    never opens the serial link / CAN bus itself) and moves the arm to a safe
    posture — so it is deferred out of bootstrap. Kinematics/IK stay local."""
    global _controller
    if _controller is not None:
        return _controller
    with _init_lock:
        if _controller is not None:
            return _controller
        from pick_place_skill.controller import PosePickPlaceController
        from pick_place_skill.primitive_clients import connect_primitives

        print("[pick_place_skill] connecting arm/hand primitives via atlas ...",
              flush=True)
        arm, hand = connect_primitives(skill)
        print("[pick_place_skill] initialising controller ...", flush=True)
        _controller = PosePickPlaceController(
            robot=arm,
            hand=hand,
            urdf_path=os.environ.get("PICK_PLACE_URDF", ""),
        )
        print("[pick_place_skill] controller ready", flush=True)
        return _controller


# ---------------------------------------------------------------------------
# Coordinate parsing
# ---------------------------------------------------------------------------


def _parse_coords(text: str, n_min: int, n_max: int):
    """Parse a comma-separated coordinate string into a list of floats.

    Accepts both ASCII and full-width commas. Returns a list of length in
    ``[n_min, n_max]``. Raises ValueError on empty / non-numeric / wrong count.
    """
    text = (text or "").strip().replace("，", ",")
    if not text:
        raise ValueError("empty coordinate")
    parts = [p for p in text.split(",") if p.strip() != ""]
    if not (n_min <= len(parts) <= n_max):
        raise ValueError(f"expected {n_min}-{n_max} numbers, got {len(parts)}")
    return [float(p) for p in parts]


def _resolve_pick_place(pick: str, offset: str):
    """Turn the two tool arguments into absolute base-frame ``(pick_xyz,
    place_xyz)``. ``pick`` is "x,y" (Z defaults to PICK_Z) or "x,y,z"; ``offset``
    is "dx,dy" (dz=0) or "dx,dy,dz"; place = pick + offset."""
    from pick_place_skill.config import PICK_Z

    p = _parse_coords(pick, 2, 3)
    px, py = p[0], p[1]
    pz = p[2] if len(p) == 3 else PICK_Z

    d = _parse_coords(offset, 2, 3)
    dx, dy = d[0], d[1]
    dz = d[2] if len(d) == 3 else 0.0

    return (px, py, pz), (px + dx, py + dy, pz + dz)


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def pick_and_place(pick: str = "", offset: str = "") -> str:
    """从指定位置抓取物体，再按相对偏移把它放下。

    机械臂移动到 pick 指定的绝对位置抓取物体，然后移动到 pick + offset 的位置松手放下。
    不使用视觉/相机，坐标由调用方直接给定（base 坐标系，单位米）。

    Args:
        pick: 绝对抓取位置。填 "x,y"（Z 取配置的默认抓取高度 PICK_Z）或 "x,y,z"。
        offset: 相对于抓取点的放置偏移。填 "dx,dy"（同高放置）或 "dx,dy,dz"。
            放置点 = 抓取点 + 偏移。
    """
    try:
        pick_xyz, place_xyz = _resolve_pick_place(pick, offset)
    except ValueError as exc:
        return json.dumps(
            {"ok": False, "error": f"bad coordinate: {exc}", "pick": pick, "offset": offset},
            ensure_ascii=False,
        )
    try:
        ctrl = _ensure_controller()
        result = ctrl.pick_and_place(pick_xyz, place_xyz)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


@mcp.tool()
async def move_home() -> str:
    """张开灵巧手并把机械臂移到安全位置（让开工作区）。"""
    try:
        ctrl = _ensure_controller()
        ctrl.move_home()
        return json.dumps({"ok": True, "action": "move_home"}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Capability metadata (declared to atlas after bootstrap)
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "pick_and_place",
        "description": "从 pick 指定的绝对位置抓取物体，再移动到 pick + offset 处松手放下（无视觉，坐标直接给定，base 系、米）。"
        "pick 填 'x,y'（Z 取默认抓取高度）或 'x,y,z'；offset 填 'dx,dy'（同高）或 'dx,dy,dz'。",
        "input_schema": {
            "type": "object",
            "properties": {
                "pick": {
                    "type": "string",
                    "description": "绝对抓取位置：'x,y' 或 'x,y,z'（base 坐标系，米）",
                },
                "offset": {
                    "type": "string",
                    "description": "相对抓取点的放置偏移：'dx,dy' 或 'dx,dy,dz'，放置点=抓取点+偏移",
                },
            },
            "required": ["pick", "offset"],
        },
    },
    {
        "name": "move_home",
        "description": "张开灵巧手并把机械臂移到安全位置（让开工作区）。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # 1. Attach the MCP app before bootstrap.
    skill.use_mcp_app(mcp)

    # 2. Bootstrap: register with atlas, start gRPC + MCP servers, heartbeat.
    print("[pick_place_skill] bootstrapping...", flush=True)
    skill.bootstrap()
    print(f"[pick_place_skill] MCP server on port {skill._mcp_port}", flush=True)

    # 3. Declare each MCP tool as a capability.
    endpoint = skill.mcp_endpoint
    for tool in _TOOLS:
        try:
            skill.declare_mcp(
                contract_id=f"{_NAMESPACE}/{tool['name']}",
                endpoint=endpoint,
                input_schema_json=json.dumps(tool["input_schema"]),
                description=tool["description"],
            )
            print(f"[pick_place_skill] declared MCP capability: {tool['name']}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[pick_place_skill] declare {tool['name']} failed: {exc}",
                  flush=True)

    # 4. Serve until SIGTERM/SIGINT; release hardware on the way out.
    print("[pick_place_skill] ready", flush=True)
    import signal

    stop = threading.Event()

    def _on_signal(signum, _frame):
        print(f"[pick_place_skill] signal {signum}, shutting down", flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    stop.wait()

    if _controller is not None:
        try:
            _controller.shutdown()
        except Exception:  # noqa: BLE001
            traceback.print_exc()


if __name__ == "__main__":
    main()
