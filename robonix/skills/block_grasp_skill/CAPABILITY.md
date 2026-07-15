# Block Grasp Skill

YOLO-OBB detection + D1 dexterous-hand block grasping, exposed to Robonix
Pilot as MCP tools. Wraps `block_grasp/grasp_controller.py` in the
Beingbeyond_D1 repo.

## Tools

- `grasp_block(class_name: str = "", position: str = "")` — detect blocks,
  grasp one, and place it at `position`. `class_name` (e.g. `red_cube`) grasps
  the highest-score block of that class; empty grasps the highest-score block
  not yet at its colour spot. `position` is a named spot (`中间`, a colour name)
  or an `"x,y"` coordinate; empty uses the block's own colour spot
  (`PLACE_POSITIONS`). A block already within `PLACE_DISTANCE_THRESHOLD` of its
  target is skipped (`grasped: false`, `ok: true`).
  Returns JSON: `{ok, detected, grasped, class, place, message}`.
- `stack_blocks(mover_class: str = "", base_class: str = "")` — stack one block
  onto another. Give both `mover_class`/`base_class` to stack that colour pair
  (error if either colour is missing); leave both empty to pick by proximity
  (base = block nearest `STACK_POSITION`, other placed on top). Runs once; call
  `reset_stack` to run again.
- `reset_stack()` — clear stacking state.
- `move_home()` — open the hand and park the arm clear of the camera.

## Architecture

The skill is a pure robonix **consumer**: it never opens hardware itself. On the
first tool call it discovers and connects the D1 primitives over gRPC and drives
the grasp pipeline through their contracts:

- `d1_arm` — `arm/move_joint` (6 arm joints), `arm/get_state`, `arm/set_head`
  (position the head to the hand-eye calibration pose). The skill keeps its own
  IK/FK (pure compute); only joint commands/reads cross the wire.
- `d1_hand` — `hand/move_joint` (open/grasp finger poses), `hand/get_state`
  (grasp-close confirmation).
- `d1_camera` — `camera/snapshot` (one RGB frame; grasping uses 2D homography,
  no depth).

The three primitives must be deployed alongside the skill — see the
`primitive:` block in `../robonix_manifest.yaml`. Hardware parameters (serial
dev, CAN iface, camera resolution) are set there as each primitive's `config:`,
NOT via skill env vars.

## Hardware / environment

- D1 head-arm (serial) + Linker dexterous hand (CAN) + head-mounted RealSense,
  owned by the `d1_arm` / `d1_hand` / `d1_camera` primitives.
- Requires hand-eye calibration `block_grasp/handeye_calib.npz` and a YOLO-OBB
  weight under `object_detect/runs/*/weights/best.pt`.
- Runs on a Python 3.10 env with the grasp stack + robonix/mcp deps
  (see `../env_setup.sh`).

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| BEINGBEYOND_PATH | $HOME/Beingbeyond_D1 | Repo root (import path) |
| BLOCK_GRASP_PYTHON | .../envs/bb_d1_robonix/bin/python3 | Python 3.10 interpreter |
| BLOCK_GRASP_MODEL | auto (newest best.pt) | YOLO-OBB weight path |
| BLOCK_GRASP_URDF | (SDK default) | URDF for local IK/FK |
| BLOCK_GRASP_DEVICE | auto | Torch device for YOLO |
| ROBONIX_ATLAS | 127.0.0.1:50051 | Atlas control plane |

Hardware I/O parameters (hand side/CAN, arm serial dev/baud, camera resolution)
live in the primitives' `config:` in `../robonix_manifest.yaml`. Tuning constants
(Z offsets, IK, hand poses, head calibration pose, stack position) live in
`block_grasp/config.py`.
