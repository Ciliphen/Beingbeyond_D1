#!/usr/bin/env bash
# Start the D1 head camera primitive.
#
# Runs on the same Python 3.10 env as the grasp skill (has pyrealsense2 +
# beingbeyond stack AND robonix_api / mcp / grpcio). See ../../env_setup.sh.
set -euo pipefail
echo "[d1_camera] starting..."

PKG_ROOT="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG_ROOT"

# Beingbeyond_D1 repo root — vision.py (RealSenseCamera) lives there.
export BEINGBEYOND_PATH="${BEINGBEYOND_PATH:-$HOME/Beingbeyond_D1}"

# Python 3.10 env with the full stack (override BLOCK_GRASP_PYTHON if needed).
PYTHON="${BLOCK_GRASP_PYTHON:-$HOME/miniconda3/envs/bb_d1_robonix/bin/python3}"

# robonix_api ships as a source dir; the Primitive constructor then adds
# rbnx-build/codegen/{proto_gen,robonix_mcp_types} to sys.path itself.
ROBONIX_API="$(rbnx path robonix-api)"
export PYTHONPATH="${ROBONIX_API}:${PKG_ROOT}:${BEINGBEYOND_PATH}:${PYTHONPATH:-}"

exec "$PYTHON" -m d1_camera.node
