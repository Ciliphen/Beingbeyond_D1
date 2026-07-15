# SPDX-License-Identifier: MulanPSL-2.0
"""Block Grasp Skill Node.

Wraps the existing YOLO-OBB + D1 dexterous-hand grasping pipeline
(``block_grasp/grasp_controller.py``) as a Robonix Skill exposing MCP tools,
so Pilot can trigger grasping with natural language via ``rbnx chat``.

Tools:
  - grasp_block(class_name, position)        — grasp one block, place at position (name/coord/colour default)
  - stack_blocks(mover_class, base_class)    — stack one block onto another
  - reset_stack()                            — clear stacking state so stack_blocks can run again
  - move_home()                              — open hand + park arm clear of the camera

Follows the roboarm_skill pattern: a raw FastMCP app attached to the Skill,
capabilities declared manually after bootstrap. Hardware (arm, hand, camera,
YOLO) is initialised lazily on first tool call so bootstrap stays fast.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback

# -- Make the Beingbeyond_D1 repo importable (block_grasp / object_detect) --
_ROOT = os.environ.get("BEINGBEYOND_PATH", "$HOME/Beingbeyond_D1")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mcp.server.fastmcp import FastMCP
from robonix_api import Skill

# ---------------------------------------------------------------------------
# Skill provider + MCP app
# ---------------------------------------------------------------------------

_NAMESPACE = "robonix/skill/block_grasp"

skill = Skill(id="block_grasp", namespace=_NAMESPACE)
mcp = FastMCP("block-grasp")

# Lazy-initialised controller (built on first tool call → fast bootstrap).
_controller = None
_init_lock = threading.Lock()


def _find_best_model() -> str:
    """Return the newest ``object_detect/runs/*/weights/best.pt`` under the
    repo, or a conventional fallback path. Mirrors run_block_grasp.py."""
    runs_dir = os.path.join(_ROOT, "object_detect", "runs")
    fallback = os.path.join(runs_dir, "train", "weights", "best.pt")
    if not os.path.isdir(runs_dir):
        return os.environ.get("BLOCK_GRASP_MODEL", fallback)
    candidates = []
    for name in os.listdir(runs_dir):
        weights = os.path.join(runs_dir, name, "weights", "best.pt")
        if os.path.isfile(weights):
            candidates.append((os.path.getmtime(weights), weights))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return os.environ.get("BLOCK_GRASP_MODEL", fallback)


def _ensure_controller():
    """Late-init the grasp controller on first use. This opens the camera,
    the head-arm serial link, the dexterous-hand CAN bus, loads YOLO, and
    moves the arm to a safe posture — so it is deliberately deferred out of
    bootstrap. Hardware/env parameters come from environment variables."""
    global _controller
    if _controller is not None:
        return _controller
    with _init_lock:
        if _controller is not None:
            return _controller
        from block_grasp.grasp_controller import BlockGraspController

        model_path = os.environ.get("BLOCK_GRASP_MODEL") or _find_best_model()
        print(f"[block_grasp_skill] initialising controller (model={model_path})",
              flush=True)
        _controller = BlockGraspController(
            model_path=model_path,
            hand_type=os.environ.get("BLOCK_GRASP_HAND_TYPE", "right"),
            hand_can=os.environ.get("BLOCK_GRASP_HAND_CAN", "can0"),
            arm_dev=os.environ.get("BLOCK_GRASP_ARM_DEV", "/dev/ttyUSB0"),
            arm_baud=int(os.environ.get("BLOCK_GRASP_ARM_BAUD", "1000000")),
            urdf_path=os.environ.get("BLOCK_GRASP_URDF", ""),
            cam_width=int(os.environ.get("BLOCK_GRASP_CAM_WIDTH", "1280")),
            cam_height=int(os.environ.get("BLOCK_GRASP_CAM_HEIGHT", "720")),
            cam_fps=int(os.environ.get("BLOCK_GRASP_CAM_FPS", "30")),
            device=os.environ.get("BLOCK_GRASP_DEVICE", ""),
            headless=True,
            auto_grasp=False,
            show_depth=False,
        )
        print("[block_grasp_skill] controller ready", flush=True)
        return _controller


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


def _parse_position(position: str):
    """Parse the position argument: "" → None (colour default); "x,y" → [x, y]
    coordinate; anything else → a named position string."""
    position = position.strip()
    if not position:
        return None
    parts = position.replace("，", ",").split(",")
    if len(parts) == 2:
        try:
            return [float(parts[0]), float(parts[1])]
        except ValueError:
            pass
    return position


@mcp.tool()
async def grasp_block(class_name: str = "", position: str = "") -> str:
    """按颜色抓取一个积木，放到指定位置。

    通过头部相机取图，运行 YOLO-OBB 检测桌面积木，抓起一个积木后放到 position
    指定的位置。若目标积木已在目标位置附近（PLACE_DISTANCE_THRESHOLD 内），则视为
    已就位，不再抓取（返回 grasped=false, ok=true）。

    Args:
        class_name: 可选，指定只抓某个颜色/类别（如 "red_cube"），取该类最高分的；
            留空则抓画面中尚未就位、置信度最高的一个。
        position: 可选，放置位置。可填命名位置（如 "中间"、"red_cube"）或坐标
            字符串（如 "0.2,0.15"）；留空则放到该积木自身颜色对应的位置。
    """
    try:
        ctrl = _ensure_controller()
        result = ctrl.grasp_once(
            class_name=class_name or None, position=_parse_position(position)
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return json.dumps(
            {"ok": False, "error": str(exc), "class_name": class_name},
            ensure_ascii=False,
        )


@mcp.tool()
async def stack_blocks(mover_class: str = "", base_class: str = "") -> str:
    """把一个积木叠到另一个积木上（执行一次）。

    需要画面里至少有 2 个积木。可指定颜色对：把 mover_class 叠到 base_class 上
    （两者取各自最高分的块，任一颜色没检测到就报错）。两个参数须同时给或同时留空。
    留空时按就近选择：离参考点最近的作底座，另一块抓起叠到其正上方。
    完成后需先调用 reset_stack 才能再次堆叠。

    Args:
        mover_class: 可选，被抓起叠上去的积木颜色（如 "red_cube"）。
        base_class:  可选，作为底座的积木颜色（如 "blue_cube"）。
    """
    try:
        ctrl = _ensure_controller()
        result = ctrl.stack_once(
            mover_class=mover_class or None, base_class=base_class or None
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


@mcp.tool()
async def reset_stack() -> str:
    """重置堆叠状态，使 stack_blocks 可以重新执行一次。"""
    try:
        ctrl = _ensure_controller()
        ctrl.reset_stack()
        return json.dumps({"ok": True, "action": "reset_stack"}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


@mcp.tool()
async def move_home() -> str:
    """张开灵巧手并把机械臂移到安全位置（让开相机视野）。"""
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
        "name": "grasp_block",
        "description": "按颜色抓取一个积木，放到指定位置。"
        "class_name 指定只抓该颜色的最高分块（留空抓置信度最高的一个）；"
        "position 指定放置位置，可填命名位置（如 '中间'、'red_cube'）或坐标 '0.2,0.15'，留空放到该积木自身颜色位置。",
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {
                    "type": "string",
                    "description": "可选，积木颜色/类别，如 'red_cube'、'blue_cube'；留空抓任意一个",
                },
                "position": {
                    "type": "string",
                    "description": "可选，放置位置：命名位置（'中间'、'red_cube' 等）或坐标 'x,y'；留空放到该积木颜色对应位置",
                },
            },
            "required": [],
        },
    },
    {
        "name": "stack_blocks",
        "description": "把一个积木叠到另一个积木上（执行一次）。画面需至少 2 个积木。"
        "可指定颜色对（mover_class 叠到 base_class 上，须同时给），留空则就近选底座。"
        "完成后需先 reset_stack 才能再堆。",
        "input_schema": {
            "type": "object",
            "properties": {
                "mover_class": {
                    "type": "string",
                    "description": "可选，被叠上去的积木颜色，如 'red_cube'；须与 base_class 同时给",
                },
                "base_class": {
                    "type": "string",
                    "description": "可选，作底座的积木颜色，如 'blue_cube'；须与 mover_class 同时给",
                },
            },
            "required": [],
        },
    },
    {
        "name": "reset_stack",
        "description": "重置堆叠状态，使 stack_blocks 可以重新执行一次。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "move_home",
        "description": "张开灵巧手并把机械臂移到安全位置（让开相机视野）。",
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
    print("[block_grasp_skill] bootstrapping...", flush=True)
    skill.bootstrap()
    print(f"[block_grasp_skill] MCP server on port {skill._mcp_port}", flush=True)

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
            print(f"[block_grasp_skill] declared MCP capability: {tool['name']}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[block_grasp_skill] declare {tool['name']} failed: {exc}",
                  flush=True)

    # 4. Serve until SIGTERM/SIGINT; release hardware on the way out.
    print("[block_grasp_skill] ready", flush=True)
    import signal

    stop = threading.Event()

    def _on_signal(signum, _frame):
        print(f"[block_grasp_skill] signal {signum}, shutting down", flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    stop.wait()

    if _controller is not None:
        try:
            _controller._shutdown()
        except Exception:  # noqa: BLE001
            traceback.print_exc()


if __name__ == "__main__":
    main()
