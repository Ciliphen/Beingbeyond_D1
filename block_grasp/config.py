#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
Grasp configuration constants.

Mirrors the grasp-related portion of roboarm's ``config.yaml``.
All units are SI (metres, radians) unless noted otherwise.
"""
from __future__ import annotations

import os

# ── Calibration ────────────────────────────────────────────────────────────
CALIB_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "handeye_calib.npz"
)
# Camera resolution during calibration — detection pixels are scaled to match.
# If you change camera resolution, pixel coords are auto-scaled.
CALIB_CAM_WIDTH: int = 1280
CALIB_CAM_HEIGHT: int = 720

# ── Head calibration (matches calibrate_handeye.py / test_click_goto.py) ─────
HEAD_YAW_DEG: float = -10.0
HEAD_PITCH_DEG: float = 35.0

# ── YOLO detection ────────────────────────────────────────────────────────
CONF_THRESHOLD: float = 0.85       # confidence threshold for detections
IOU_THRESHOLD: float = 0.45        # IoU threshold for NMS

# ── Block geometry ──────────────────────────────────────────────────────────
BLOCK_SIZE: float = 0.05               # cube side length (m)
# Grasp Z = z_table + GRASP_Z_OFFSET (tune this!)
# Negative = go below table surface. For a 5 cm cube, the hand needs to
# wrap around the centre (~2.5 cm above table), but finger geometry means
# the EE (wrist) must often go lower.  Start at 0 and tune downward.
GRASP_Z_OFFSET: float = 0.015          # Z offset above table for grasp (m)

# ── Gravity sag compensation ───────────────────────────────────────────────
# The arm sags under its own weight when extended.  dZ = factor × dist²
# (cantilever model).  Positive = raise target to counteract sag.
# Start at 0.02, increase if far blocks are still too low.
GRAVITY_SAG_FACTOR: float = 0.3      # m of sag per m³ of horizontal distance

# ── Motion ─────────────────────────────────────────────────────────────────
Z_SAFE: float = 0.25                # safe Z height for approach / travel (m)
APPROACH_HEIGHT_OFFSET: float = 0.10   # height above target to approach first (m)
CATCH_DELAY_S: float = 0.8            # pause after hand open/close (CAN bus latency)
INTERP_STEP_SIZE: float = 0.025       # interpolation step size for Jacobian IK (m)
MAX_DXY: float = 0.50                 # max XY distance from initial EE position (m)

# ── IK solver parameters ───────────────────────────────────────────────────
IK_Z_WEIGHT: float = 3.0              # extra weight on Z axis (>1 = prioritise height)
IK_POS_TOL: float = 0.005             # position tolerance for SLSQP final refinement (m)
IK_TILT_TOL_DEG: float = 5.0          # tilt tolerance (deg)
IK_YAW_TOL_DEG: float = 10.0          # yaw tolerance (deg)
IK_MAX_ITERS: int = 200               # max SLSQP iterations
IK_N_RESTARTS: int = 4                # multi-restart attempts
IK_FAIL_THRESHOLD: float = 0.02       # max acceptable IK error for interpolated steps (m)

# ── Near-singularity guard ──────────────────────────────────────────────────
# Reject any IK solution that jumps an arm joint (excl. j6 wrist roll) more
# than this in a single interpolation/teleop step — signals a near-singularity
# or solution-branch switch that would make the EE lurch.  Shared by
# grasp_controller and test_ee_teleop.
JOINT_JUMP_THR_DEG: float = 20.0

# ── Dexterous hand poses (6-D normalised [0, 1]; 0=open, 1=closed) ───────
# Joint order: thumb_cmc_pitch, thumb_cmc_yaw, index_mcp_pitch,
#              middle_mcp_pitch, ring_mcp_pitch, pinky_mcp_pitch
HAND_OPEN: list[float] = [0.1, 0.8, 0.1, 0.1, 0.0, 0.0]         # thumb_yaw=0.8 always
HAND_GRASP: list[float] = [0.35, 0.8, 0.40, 0.40, 0.0, 0.0]   # ~4.5 cm grip, thumb opposed
HAND_CLOSE: list[float] = [0.7, 0.5, 0.8, 0.8, 0.8, 0.8]    # max tight

# Grasp success: after closing to HAND_GRASP, the average finger position
# should be in [GRASP_OK_MIN, GRASP_OK_MAX].  If too close to OPEN → empty
# grasp.  If too close to CLOSE → nothing to block the fingers.
GRASP_OK_MIN: float = 0.20   # below this = still open → no object
GRASP_OK_MAX: float = 0.60   # above this = fully closed → nothing blocking

# ── Stacking ──────────────────────────────────────────────────────────────
# Stacking mode: pick up one block and stack it onto another.
# When no colour pair is specified, the block nearest STACK_POSITION is used
# as the base; STACK_POSITION is only a reference point for that choice.
STACK_POSITION: list[float] = [0.25, 0.0, 0.105]  # reference point for default base

# ── Place positions per colour (base-frame x, y in metres) ────────────────
# Used by the grasp_block action (grasp one block, then place it at its
# colour's spot). Place Z is derived from the measured table height, so only
# x,y are configured here. Not used by stacking.
PLACE_POSITIONS: dict[str, list[float]] = {
    "red_cube":    [0.135, -0.220],
    "blue_cube":   [0.350, -0.220],
    "green_cube":  [0.350, 0.250],
    "yellow_cube": [0.135, 0.250],
}
# A block within this XY distance of its target position is treated as already
# placed → grasp_block skips it (no re-grasp).
PLACE_DISTANCE_THRESHOLD: float = 0.05  # 5 cm

# Named drop positions addressable by name in grasp_block's position argument.
# The per-colour PLACE_POSITIONS above are also addressable by colour name.
NAMED_POSITIONS: dict[str, list[float]] = {
    "中间": [0.198, 0.107],
}

# Arm parks here after placing to clear the camera view; also the fallback
# drop spot for grasp_block when a block's colour has no place position.
ASIDE_POSITION: list[float] = [0.20, 0.0, 0.25]  # base-frame x, y, z

# ── Grasp position offset (world XY, metres) ──────────────────────────────
# Fine-tune the grasp point relative to the detected bottom-face centre.
GRASP_OFFSET_X: float = 0.0     # +X = forward (away from robot base)
GRASP_OFFSET_Y: float = 0.0      # +Y = left, -Y = right

# ── Grasp yaw compensation ────────────────────────────────────────────────
# The OBB angle is perpendicular to the long edge.  The thumb sits on the
# LEFT side of the palm (for right hand).  Tune this offset so the thumb
# wraps around the cube instead of bumping into it.
GRASP_YAW_OFFSET_DEG: float = 0.0   # add to OBB angle (try ±90 if thumb misaligned)

# ── OBB grasp point ───────────────────────────────────────────────────────
# The OBB encloses the visible projection (top + side faces).
# The grasp point lies between the box centre (0.0) and the bottom edge (1.0).
# With perspective correction enabled, the geometric offset is handled
# automatically. Set > 0 only if fine-tuning is needed.
OBB_GRASP_RATIO: float = 0.0

# ── Depth sampling ────────────────────────────────────────────────────────
DEPTH_SAMPLE_RADIUS: int = 5          # pixel radius around detection centre
                                      # for median depth estimation

# ── Target EE orientation (RPY in degrees) ────────────────────────────────
# Roll + Pitch keep the hand perpendicular to the table.
# Yaw is the default orientation; OBB angle is applied as a delta on top.
EE_ROLL_DEG: float = 178.0
EE_PITCH_DEG: float = 61.0
EE_YAW_DEG: float = -175.0
