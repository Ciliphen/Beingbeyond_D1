#!/usr/bin/env bash
# Start the soma_d1 body-model gRPC service.
#
# Runs on the same Python 3.10 env as the grasp skill (has beingbeyond_d1_sdk
# for the bundled D1 URDF, plus grpcio/robonix_api). See ../env_setup.sh.
set -euo pipefail
echo "[soma_d1] starting..."

PKG_ROOT="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG_ROOT"

PYTHON="${BLOCK_GRASP_PYTHON:-$HOME/miniconda3/envs/bb_d1_robonix/bin/python3}"
ROBONIX_API="$(rbnx path robonix-api)"

# gRPC stubs (soma_pb2, geometry_msgs_pb2, robonix_contracts_pb2_grpc) live
# in the codegen output; put it + robonix-api + the package dir on PYTHONPATH.
export PYTHONPATH="$PKG_ROOT/rbnx-build/codegen/proto_gen:${ROBONIX_API}:${PKG_ROOT}:${PYTHONPATH:-}"

exec "$PYTHON" -m soma_d1.service
