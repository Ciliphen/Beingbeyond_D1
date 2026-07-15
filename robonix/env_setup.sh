#!/usr/bin/env bash
# Create the Python 3.10 conda env that runs the block_grasp Robonix skill.
#
# Why a new env: the default `bb_d1` env is Python 3.8, but robonix_api + mcp +
# fastmcp require Python >= 3.10. The D1 SDK ships a cp310 wheel, so a 3.10 env
# can hold BOTH the grasp hardware stack AND the robonix skill deps.
#
# Run this yourself (needs network). Re-runnable.
set -euo pipefail

ENV_NAME="${ENV_NAME:-bb_d1_robonix}"
REPO="${BEINGBEYOND_PATH:-$HOME/Beingbeyond_D1}"
SDK_WHEEL="${REPO}/lib/beingbeyond_d1_sdk-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "[env_setup] conda env '${ENV_NAME}' already exists, reusing"
else
    echo "[env_setup] creating conda env '${ENV_NAME}' (python 3.10)"
    conda create -y -n "${ENV_NAME}" python=3.10
fi

# Resolve the env's python without needing `conda activate` in a script.
ENV_PY="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"
echo "[env_setup] env python: ${ENV_PY}"

echo "[env_setup] installing D1 SDK (cp310 wheel)"
[ -f "${SDK_WHEEL}" ] || { echo "SDK wheel not found: ${SDK_WHEEL}" >&2; exit 1; }
"${ENV_PY}" -m pip install "${SDK_WHEEL}"

echo "[env_setup] installing pinned deps from requirements.txt"
REQ="$(dirname "$(readlink -f "$0")")/requirements.txt"
[ -f "${REQ}" ] || { echo "requirements.txt not found: ${REQ}" >&2; exit 1; }
"${ENV_PY}" -m pip install -r "${REQ}"

echo "[env_setup] verifying imports"
"${ENV_PY}" - <<'PY'
import importlib.util
mods = ["beingbeyond_d1_sdk", "pyrealsense2", "ultralytics", "cv2", "scipy",
        "mcp", "fastmcp", "grpc", "grpc_tools"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("missing:", missing or "none")
raise SystemExit(1 if missing else 0)
PY

echo "[env_setup] done. Set BLOCK_GRASP_PYTHON=${ENV_PY} (or use the default env path in start.sh)."
