#!/usr/bin/env bash
# Build phase: run rbnx codegen so the camera gRPC stubs + MCP dataclasses
# (sensor_msgs_mcp, std_msgs_mcp, builtin_interfaces_mcp) are generated under
# rbnx-build/codegen/.
set -euo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# rbnx codegen invokes `python3` from PATH and needs grpcio-tools there. Put
# the 3.10 env (env_setup.sh installs grpcio-tools into it) first.
ENV_PY="${BLOCK_GRASP_PYTHON:-$HOME/miniconda3/envs/bb_d1_robonix/bin/python3}"
[ -x "$ENV_PY" ] && export PATH="$(dirname "$ENV_PY"):$PATH"

# --mcp: depth_snapshot is still an MCP contract, so generate the MCP
# dataclasses (sensor_msgs_mcp, std_msgs_mcp, builtin_interfaces_mcp) too.
# snapshot is served over gRPC (proto_gen) for the block_grasp skill.
FLAGS=(--mcp)
[[ "${RBNX_BUILD_CLEAN:-}" == "1" ]] && FLAGS+=(--clean)
rbnx codegen -p "$PKG" "${FLAGS[@]}"
echo "[d1_camera] build done"
