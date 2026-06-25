#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
set -euo pipefail
echo "[d1_hand] build …"
rbnx codegen --mcp
echo "[d1_hand] build done"
