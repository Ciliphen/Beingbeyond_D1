#!/usr/bin/env bash
# Build phase: run rbnx codegen so the soma gRPC servicer stubs + message
# classes (soma_pb2, geometry_msgs_pb2) are generated under rbnx-build/.
set -euo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# rbnx codegen invokes `python3` from PATH and needs grpcio-tools there. Put
# the 3.10 env (env_setup.sh installs grpcio-tools into it) first.
ENV_PY="${BLOCK_GRASP_PYTHON:-/home/xlf/miniconda3/envs/bb_d1_robonix/bin/python3}"
[ -x "$ENV_PY" ] && export PATH="$(dirname "$ENV_PY"):$PATH"

rbnx codegen -p "$PKG"
echo "[soma_d1] build done"
