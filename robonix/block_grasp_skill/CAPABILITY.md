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

## Hardware / environment

- D1 head-arm (serial) + Linker dexterous hand (CAN) + head-mounted RealSense.
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
| BLOCK_GRASP_HAND_TYPE | right | Dexterous hand side |
| BLOCK_GRASP_HAND_CAN | can0 | Hand CAN interface |
| BLOCK_GRASP_ARM_DEV | /dev/ttyUSB0 | Head-arm serial device |
| BLOCK_GRASP_ARM_BAUD | 1000000 | Head-arm serial baud |
| BLOCK_GRASP_CAM_WIDTH / _HEIGHT / _FPS | 1280 / 720 / 30 | Camera config |
| ROBONIX_ATLAS | 127.0.0.1:50051 | Atlas control plane |

Tuning constants (Z offsets, IK, hand poses, stack position) live in
`block_grasp/config.py`.
