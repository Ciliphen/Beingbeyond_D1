# SPDX-License-Identifier: MulanPSL-2.0
"""Tunables specific to the pick_place skill.

Everything motion-related (EE orientation, approach height, gravity-sag, IK
tolerances, hand poses, grasp-success band, park pose) is reused verbatim from
``block_grasp.config`` — this skill only adds the one constant the block-grasp
pipeline did not need: a default grasp height for when the caller gives a 2D
(x, y) pick target without an explicit Z.

There is no camera / hand-eye calibration in this skill, so the table height is
a fixed, hand-measured constant rather than something read from a calib file.
Measure it once for your table and set ``PICK_Z`` here (base-frame metres).
"""
from __future__ import annotations

# Default grasp height (base-frame Z, metres) used when a pick target is given
# as "x,y" without an explicit Z. Roughly the cube-centre height above the
# table for the D1 desktop setup — retune per table / object.
PICK_Z: float = 0.095
