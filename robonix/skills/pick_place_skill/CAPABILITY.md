# Pick-Place Skill

Coordinate-based pick-and-place for the D1 arm + dexterous hand, exposed to
Robonix Pilot as MCP tools. Grasps at an absolute position and releases at a
position given relative to it. **No vision** (no camera, no YOLO, no hand-eye
calibration) — coordinates are supplied directly. Wraps
`pick_place_skill/controller.py`, reusing the kinematics / IK / motion tunables
from `block_grasp/` in the Beingbeyond_D1 repo.

## Tools

- `pick_and_place(pick: str, offset: str)` — grasp the object at absolute `pick`,
  then move to `pick + offset` and release. Both are base-frame coordinates in
  metres. `pick` is `"x,y"` (Z defaults to the configured `PICK_Z` grasp height)
  or `"x,y,z"`; `offset` is `"dx,dy"` (place at the same height) or `"dx,dy,dz"`.
  Grasp success is checked from the hand's finger closure; if it fails the place
  is skipped. Returns JSON: `{ok, grasped, placed, pick, place, message}`.
- `move_home()` — open the hand and park the arm clear of the workspace.

## Architecture

The skill is a pure robonix **consumer**: it never opens hardware itself. On the
first tool call it discovers and connects the D1 primitives over gRPC and drives
the pick/place through their contracts:

- `d1_arm` — `arm/move_joint` (6 arm joints), `arm/get_state`. IK/FK stay local
  (pure compute); only joint commands/reads cross the wire. The head is **not**
  touched — the end-effector pose depends only on the arm joints, so there is no
  `set_head` and no camera alignment.
- `d1_hand` — `hand/move_joint` (open/grasp finger poses), `hand/get_state`
  (grasp-close confirmation).

The two primitives must be deployed alongside the skill — see the `primitive:`
block in `../robonix_manifest.yaml`. Hardware parameters (serial dev, CAN iface)
are set there as each primitive's `config:`, NOT via skill env vars.

## Hardware / environment

- D1 head-arm (serial) + Linker dexterous hand (CAN), owned by the `d1_arm` /
  `d1_hand` primitives. No camera is used.
- Runs on a Python 3.10 env with the D1 SDK/kinematics stack + robonix/mcp deps
  (see `../env_setup.sh`).

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| BEINGBEYOND_PATH | $HOME/Beingbeyond_D1 | Repo root (import path) |
| PICK_PLACE_PYTHON | .../envs/bb_d1_robonix/bin/python3 | Python 3.10 interpreter |
| PICK_PLACE_URDF | (SDK default) | URDF for local IK/FK |
| ROBONIX_ATLAS | 127.0.0.1:50051 | Atlas control plane |

Hardware I/O parameters (hand side/CAN, arm serial dev/baud) live in the
primitives' `config:` in `../robonix_manifest.yaml`. The default grasp height
`PICK_Z` (used when a pick target omits Z) lives in `pick_place_skill/config.py`;
all other motion tunables (EE orientation, approach height, gravity sag, IK
tolerances, hand poses, grasp-success band, park pose) are reused from
`block_grasp/config.py`.
