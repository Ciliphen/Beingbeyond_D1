#!/usr/bin/env bash
# Start the D1 arm primitive.
#
# Runs on the same Python 3.10 env as the grasp skill (has the beingbeyond_d1_sdk
# cp310 wheel + numpy/scipy AND robonix_api / grpcio). See ../../env_setup.sh.
#
# NOTE: the HeadArm chain (arm + head) shares one serial device (/dev/ttyUSB0).
# Do NOT run this at the same time as the block_grasp skill or any other process
# that opens the same port — they will contend for the serial link.
set -euo pipefail
echo "[d1_arm] starting..."

PKG_ROOT="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG_ROOT"

# Python 3.10 env with the D1 SDK + robonix deps (override BLOCK_GRASP_PYTHON).
PYTHON="${BLOCK_GRASP_PYTHON:-$HOME/miniconda3/envs/bb_d1_robonix/bin/python3}"

# robonix_api ships as a source dir; the Primitive constructor then adds
# rbnx-build/codegen/{proto_gen,robonix_mcp_types} to sys.path itself.
ROBONIX_API="$(rbnx path robonix-api)"
export PYTHONPATH="${ROBONIX_API}:${PKG_ROOT}:${PYTHONPATH:-}"

exec "$PYTHON" -m d1_arm.node
