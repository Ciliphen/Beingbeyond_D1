#!/usr/bin/env bash
# Build phase for the block_grasp skill: run rbnx codegen so the Driver
# lifecycle gRPC stubs are generated. MCP tools are declared at runtime,
# so there are no contract files to compile here.
set -euo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# rbnx codegen invokes `python3` from PATH to emit the _pb2 stubs and needs
# grpcio-tools there. Put the 3.10 env (which env_setup.sh installs
# grpcio-tools into) first so codegen uses it rather than base python.
ENV_PY="${BLOCK_GRASP_PYTHON:-$HOME/miniconda3/envs/bb_d1_robonix/bin/python3}"
[ -x "$ENV_PY" ] && export PATH="$(dirname "$ENV_PY"):$PATH"

rbnx codegen -p "$PKG"
echo "[block_grasp_skill] build done"
