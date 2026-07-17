#!/usr/bin/env bash
# Start the pick_place skill node.
#
# Runs on a dedicated Python 3.10 env that has BOTH the D1 SDK/kinematics stack
# (beingbeyond_d1_sdk cp310 wheel, scipy) AND the robonix skill deps
# (robonix_api, mcp, fastmcp, grpcio). The default bb_d1 env is Python 3.8 and
# cannot run robonix_api/mcp — see env_setup.sh.
set -euo pipefail
echo "[pick_place_skill] starting..."

# Beingbeyond_D1 repo root (block_grasp kinematics/ik/config live here).
export BEINGBEYOND_PATH="${BEINGBEYOND_PATH:-$HOME/Beingbeyond_D1}"

# Python 3.10 env with the full stack (override PICK_PLACE_PYTHON if needed).
PYTHON="${PICK_PLACE_PYTHON:-$HOME/miniconda3/envs/bb_d1_robonix/bin/python3}"

# robonix_api is served from the robonix source tree, not pip-installed.
ROBONIX_API="$(rbnx path robonix-api)"
export PYTHONPATH="${ROBONIX_API}:${BEINGBEYOND_PATH}:${PYTHONPATH:-}"

exec "$PYTHON" -m pick_place_skill.node
