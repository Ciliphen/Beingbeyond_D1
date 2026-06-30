#!/usr/bin/env python3
"""
Custom Jacobian-based IK solver for D1 arm (6-DOF).

Uses damped least-squares (Levenberg-Marquardt) with task-space error weighting
and joint-limit nullspace projection.  More robust than the SDK's QP optimizer
for pure-Z-plane moves because:
  - No competing posture objective pulling Z away
  - Higher Z weight ensures the solver prioritises Z accuracy
  - Adaptive damping handles near-singular configurations gracefully
  - Multi-restart avoids local minima

Usage:
    from block_grasp.ik_jacobian import jacobian_ik, jacobian_ik_multi_restart
    q_head, q_arm, err, iters = jacobian_ik(kin, T_target, q_head, q_arm_init)
"""

from typing import Optional, Tuple

import numpy as np

# Lazy import – pinocchio is heavy
_pin = None

def _get_pin():
    global _pin
    if _pin is None:
        import pinocchio as pin
        _pin = pin
    return _pin


# ── Arm joint indices in the full 19-DOF q ──────────────────────────
ARM_IDX = slice(2, 8)   # [joint_1 .. joint_6]


def jacobian_ik(
    kin,                        # D1Kinematics
    T_target: np.ndarray,       # (4,4) target SE(3) in base frame
    q_head: np.ndarray,         # (2,) head joints (fixed)
    q_arm_init: np.ndarray,     # (6,) arm initial guess
    *,
    pos_weight: float = 1.0,    # overall position-error multiplier
    z_weight: float = 1.0,      # extra Z-axis weight (>1 = prioritise height)
    ori_weight: float = 1.0,    # orientation-error multiplier
    damping: float = 0.05,      # initial LM damping (adaptively reduced)
    min_damping: float = 1e-6,
    max_iters: int = 300,
    tol_pos: float = 1e-4,      # position convergence (m)
    tol_ori: float = 1e-3,      # orientation convergence (rad)
    dt: float = 0.8,            # step size (<1 = conservative, 1 = full step)
    line_search: bool = True,   # whether to back-track on error increase
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Damped least-squares Jacobian IK (arm-only, head fixed).

    Returns (q_head, q_arm, err_norm, iterations).
    q_head is returned unchanged (for API compatibility with SDK IK).
    """
    pin = _get_pin()
    k = kin._kin
    m = k.model
    d = k.data

    ee_fid = k.ee_frame_id
    lower = k.lower_limits[ARM_IDX].copy()
    upper = k.upper_limits[ARM_IDX].copy()

    q_arm = q_arm_init.astype(float).copy()
    q_full = kin.make_q(q_head, q_arm)

    # ── Build task-space weighting matrix ────────────────────────────
    W = np.eye(6)
    W[0, 0] = pos_weight         # x
    W[1, 1] = pos_weight         # y
    W[2, 2] = pos_weight * z_weight  # z — higher priority
    W[3, 3] = ori_weight         # rx
    W[4, 4] = ori_weight         # ry
    W[5, 5] = ori_weight         # rz

    lam = damping
    best_err = np.inf
    best_q_arm = q_arm.copy()

    for it in range(max_iters):
        # Forward kinematics
        pin.forwardKinematics(m, d, q_full)
        pin.updateFramePlacements(m, d)

        M_cur = d.oMf[ee_fid]               # SE3 object in base frame
        # Pose error: twist taking M_cur → T_target, expressed in LOCAL (EE) frame
        M_target = pin.SE3(T_target)        # convert np array → SE3
        M_err = M_cur.actInv(M_target)      # = M_cur^{-1} * M_target
        err = pin.log6(M_err).vector        # (6,) [vx,vy,vz, wx,wy,wz]

        pos_err = np.linalg.norm(err[:3])
        ori_err = np.linalg.norm(err[3:])

        if pos_err < tol_pos and ori_err < tol_ori:
            best_q_arm = q_arm.copy()
            best_err = pos_err + ori_err
            break

        # Jacobian — LOCAL frame (matches M_err frame)
        J = pin.computeFrameJacobian(m, d, q_full, ee_fid, pin.ReferenceFrame.LOCAL)
        J_arm = J[:, ARM_IDX]          # (6, 6) arm sub-Jacobian

        # Weighted damped pseudoinverse:  dq = J^T (J J^T + λ² W^{-1})^{-1} err
        # Equivalently:  dq = J^T solve(J J^T + λ²I,  W @ err)
        JJt = J_arm @ J_arm.T
        A = JJt + lam * lam * np.eye(6)
        try:
            weighted_err = W @ err
            dq = J_arm.T @ np.linalg.solve(A, weighted_err) * dt
        except np.linalg.LinAlgError:
            lam *= 10
            continue

        # ── Line search (simple backtrack) ───────────────────────────
        if line_search:
            q_test = np.clip(q_arm + dq, lower, upper)
            q_full_test = kin.make_q(q_head, q_test)
            pin.forwardKinematics(m, d, q_full_test)
            pin.updateFramePlacements(m, d)
            M_test = d.oMf[ee_fid]
            M_err_test = M_test.actInv(M_target)
            err_test = pin.log6(M_err_test).vector
            new_err = np.linalg.norm(err_test[:3]) + np.linalg.norm(err_test[3:])
            old_err = pos_err + ori_err
            if new_err > old_err * 1.01 and dt > 0.1:
                dq *= 0.5   # backtrack
        else:
            new_err = pos_err + ori_err  # skip

        # Apply step with joint-limit clamping
        q_arm = np.clip(q_arm + dq, lower, upper)

        # ── Nullspace joint-limit avoidance ──────────────────────────
        # Push joints away from limits using the nullspace projector
        margin = 0.05  # rad (~3°) — start pushing when this close to limit
        q_mid = (lower + upper) / 2
        q_range = (upper - lower) / 2 + 1e-6
        # Normalised distance from mid: -1=at lower, +1=at upper
        q_norm = (q_arm - q_mid) / q_range
        # Repulsion gradient: strong near limits, zero in centre
        grad = np.zeros(6)
        for j in range(6):
            if q_norm[j] > 1 - margin / q_range[j]:
                grad[j] = -(q_norm[j] - (1 - margin / q_range[j])) / (margin / q_range[j])
            elif q_norm[j] < -1 + margin / q_range[j]:
                grad[j] = -(q_norm[j] - (-1 + margin / q_range[j])) / (margin / q_range[j])
        # Nullspace projector: N = I - J^+ J
        try:
            J_pinv = J_arm.T @ np.linalg.solve(JJt + lam * lam * np.eye(6), np.eye(6))
            N = np.eye(6) - J_pinv @ J_arm
            q_arm = np.clip(q_arm + N @ grad * 0.002, lower, upper)
        except np.linalg.LinAlgError:
            pass

        q_full = kin.make_q(q_head, q_arm)

        # Track best
        if new_err < best_err:
            best_err = new_err
            best_q_arm = q_arm.copy()

        # Adaptive damping: reduce if converging, increase if stuck
        if new_err < old_err * 0.9:
            lam = max(min_damping, lam * 0.7)
        elif new_err > old_err * 1.1:
            lam *= 2.0

    else:
        # Exceeded max_iters — return best found
        q_arm = best_q_arm

    # Final error
    q_full = kin.make_q(q_head, q_arm)
    pin.forwardKinematics(m, d, q_full)
    pin.updateFramePlacements(m, d)
    M_final = d.oMf[ee_fid]
    M_target_final = pin.SE3(T_target)
    M_err_final = M_final.actInv(M_target_final)
    final_err = pin.log6(M_err_final).vector
    final_pos = float(np.linalg.norm(final_err[:3]))
    final_ori = float(np.linalg.norm(final_err[3:]))

    return q_head.copy(), q_arm.copy(), final_pos + final_ori, it + 1


def jacobian_ik_multi_restart(
    kin,
    T_target: np.ndarray,
    q_head: np.ndarray,
    q_arm_current: np.ndarray,
    *,
    n_restarts: int = 5,
    noise_scale: float = 0.15,   # rad std for random restarts
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Multi-restart Jacobian IK.

    Tries from:
      1. current q_arm
      2–N. current q_arm + random noise

    Returns the best (lowest error) solution.
    """
    best_result = None
    best_err = np.inf

    candidates = [q_arm_current.copy()]
    rng = np.random.RandomState(42)
    for _ in range(n_restarts - 1):
        noise = rng.randn(6) * noise_scale
        candidates.append(np.clip(q_arm_current + noise,
                                  kin._kin.lower_limits[ARM_IDX],
                                  kin._kin.upper_limits[ARM_IDX]))

    for i, q0 in enumerate(candidates):
        try:
            q_h, q_a, err, it = jacobian_ik(kin, T_target, q_head, q0, **kwargs)
            if err < best_err:
                best_err = err
                best_result = (q_h, q_a, err, it)
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("jacobian_ik_multi_restart: all restarts failed")

    return best_result
