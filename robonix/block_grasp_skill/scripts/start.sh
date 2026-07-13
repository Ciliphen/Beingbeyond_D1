#!/usr/bin/env bash
# Start the block_grasp skill node.
#
# Runs on a dedicated Python 3.10 env that has BOTH the grasp hardware stack
# (beingbeyond_d1_sdk cp310 wheel, pyrealsense2, ultralytics) AND the robonix
# skill deps (robonix_api, mcp, fastmcp, grpcio). The default bb_d1 env is
# Python 3.8 and cannot run robonix_api/mcp — see env_setup.sh.
set -euo pipefail
echo "[block_grasp_skill] starting..."

# Beingbeyond_D1 repo root (block_grasp / object_detect / clients live here).
export BEINGBEYOND_PATH="${BEINGBEYOND_PATH:-/home/xlf/Beingbeyond_D1}"

# Python 3.10 env with the full stack (override BLOCK_GRASP_PYTHON if needed).
PYTHON="${BLOCK_GRASP_PYTHON:-/home/xlf/miniconda3/envs/bb_d1_robonix/bin/python3}"

# robonix_api is served from the robonix source tree, not pip-installed.
ROBONIX_API="$(rbnx path robonix-api)"
export PYTHONPATH="${ROBONIX_API}:${BEINGBEYOND_PATH}:${PYTHONPATH:-}"

exec "$PYTHON" -m block_grasp_skill.node
