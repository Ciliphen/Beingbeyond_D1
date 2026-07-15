#!/usr/bin/env bash
# Start the D1 dexterous-hand primitive.
#
# Runs on the same Python 3.10 env as the grasp skill (has the beingbeyond_d1_sdk
# cp310 wheel AND robonix_api / grpcio). See ../../env_setup.sh.
#
# NOTE: DexHand brings the CAN interface up at init (`ip link set …`). If this
# process is not root, export PREFLIGHT_SUDO_PASS so the SDK's `sudo -S` works,
# or bring can0 up beforehand.
set -euo pipefail
echo "[d1_hand] starting..."

PKG_ROOT="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG_ROOT"

# Python 3.10 env with the D1 SDK + robonix deps (override BLOCK_GRASP_PYTHON).
PYTHON="${BLOCK_GRASP_PYTHON:-$HOME/miniconda3/envs/bb_d1_robonix/bin/python3}"

# robonix_api ships as a source dir; the Primitive constructor then adds
# rbnx-build/codegen/{proto_gen,robonix_mcp_types} to sys.path itself.
ROBONIX_API="$(rbnx path robonix-api)"
export PYTHONPATH="${ROBONIX_API}:${PKG_ROOT}:${PYTHONPATH:-}"

exec "$PYTHON" -m d1_hand.node
