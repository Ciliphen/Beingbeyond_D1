#!/usr/bin/env bash
# Build phase: run rbnx codegen so the hand gRPC servicer stubs + message
# classes (hand_pb2, hand_pb2_grpc) are generated under rbnx-build/codegen/.
set -euo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# rbnx codegen invokes `python3` from PATH and needs grpcio-tools there. Put
# the 3.10 env (env_setup.sh installs grpcio-tools into it) first.
ENV_PY="${BLOCK_GRASP_PYTHON:-$HOME/miniconda3/envs/bb_d1_robonix/bin/python3}"
[ -x "$ENV_PY" ] && export PATH="$(dirname "$ENV_PY"):$PATH"

FLAGS=()
[[ "${RBNX_BUILD_CLEAN:-}" == "1" ]] && FLAGS+=(--clean)
rbnx codegen -p "$PKG" "${FLAGS[@]}"
echo "[d1_hand] build done"
