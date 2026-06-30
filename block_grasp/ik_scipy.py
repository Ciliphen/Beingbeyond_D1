#!/usr/bin/env python3
"""
scipy.optimize 驱动的 IK 求解器 — 参考 roboarm 实现。

与 roboarm arm_base._ik_cost_function 一致的设计：
  - 姿态误差分解为 tilt（EE Z轴指向）和 yaw（绕Z旋转）
  - 位置 / tilt / yaw 分别归一化加权
  - SLSQP 优化器 + URDF 关节限位
  - 当前位置作为初始猜测（避免漂移）

优势：
  - tilt/yaw 分离 → 可以放松 yaw 权重（桌面抓取绕Z旋转不重要）
  - Z 轴权重 → 优先保证 EE 高度
  - 多起点重启 → 避免局部极小

Usage:
    from block_grasp.ik_scipy import scipy_ik, scipy_ik_multi_restart
    q_head, q_arm, err, it = scipy_ik(kin, T_target, q_head, q_arm_init)
"""

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


# ── Arm joint indices in the full 19-DOF q ──────────────────────────
ARM_IDX = slice(2, 8)   # [joint_1 .. joint_6]


def _rotation_error_tilt_yaw(R_current: np.ndarray, R_target: np.ndarray):
    """
    将旋转误差分解为 tilt（Z轴夹角）和 yaw（绕Z旋转角）。

    Returns:
        tilt_rad: 当前Z轴与目标Z轴之间的角度 (0 ~ π)
        yaw_rad:  绕目标Z轴，当前X轴投影与目标X轴的夹角 (-π ~ π)
    """
    # Z 轴（EE 指向）
    z_cur = R_current[:, 2]
    z_tgt = R_target[:, 2]

    # Tilt: Z轴夹角
    cos_tilt = np.clip(np.dot(z_cur, z_tgt), -1.0, 1.0)
    tilt_rad = float(np.arccos(cos_tilt))

    # Yaw: 绕目标Z轴，把当前X轴投影到目标XY平面，与目标X轴的夹角
    # 如果 tilt 很小（Z轴基本对齐），直接用 atan2
    x_cur = R_current[:, 0]
    # 投影当前X到目标XY平面
    x_proj = x_cur - np.dot(x_cur, z_tgt) * z_tgt
    norm_proj = np.linalg.norm(x_proj)
    if norm_proj < 1e-10:
        yaw_rad = 0.0
    else:
        x_proj /= norm_proj
        x_tgt = R_target[:, 0]
        y_tgt = R_target[:, 1]
        # x_proj 在目标XY平面上的坐标
        proj_x = np.dot(x_proj, x_tgt)
        proj_y = np.dot(x_proj, y_tgt)
        yaw_rad = float(np.arctan2(proj_y, proj_x))

    return tilt_rad, yaw_rad


def _ik_cost(
    q_arm: np.ndarray,
    kin,                        # D1Kinematics
    q_head: np.ndarray,
    T_target: np.ndarray,
    pos_tol: float,
    tilt_tol: float,
    yaw_tol: float,
    z_weight: float,
) -> float:
    """IK 代价函数 — 与 roboarm 结构一致。"""
    # 正向运动学
    T_cur = kin.ee_in_base(q_head, q_arm)

    # 位置误差
    p_cur = T_cur[:3, 3]
    p_tgt = T_target[:3, 3]
    dx = p_cur[0] - p_tgt[0]
    dy = p_cur[1] - p_tgt[1]
    dz = p_cur[2] - p_tgt[2]
    pos_error = np.sqrt(dx*dx + dy*dy + dz*dz)

    # Z轴单独加权：加权平方和 = dx² + dy² + z_weight * dz²
    pos_term = (dx*dx + dy*dy + z_weight * dz*dz) / (pos_tol * pos_tol)

    # 姿态误差 (tilt / yaw 分解)
    R_cur = T_cur[:3, :3]
    R_tgt = T_target[:3, :3]
    tilt_rad, yaw_rad = _rotation_error_tilt_yaw(R_cur, R_tgt)

    tilt_term = (tilt_rad / tilt_tol) ** 2
    yaw_term = (yaw_rad / yaw_tol) ** 2

    return float(pos_term + tilt_term + yaw_term)


def scipy_ik(
    kin,                        # D1Kinematics
    T_target: np.ndarray,       # (4,4) target SE(3) in base frame
    q_head: np.ndarray,         # (2,) head joints (fixed)
    q_arm_init: np.ndarray,     # (6,) arm initial guess
    *,
    pos_tol: float = 0.01,      # 位置容差 (m) — roboarm 默认 1cm
    tilt_tol_deg: float = 5.0,  # tilt 容差 (度) — roboarm 默认 5°
    yaw_tol_deg: float = 10.0,  # yaw 容差 (度) — roboarm 默认 10°
    z_weight: float = 3.0,      # Z 轴额外权重 (>1=优先保证高度)
    max_iters: int = 200,       # SLSQP maxiter
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    SLSQP 数值 IK（arm-only, head 固定），参考 roboarm 实现。

    Returns (q_head, q_arm, final_cost, iterations).
    """
    lower = kin._kin.lower_limits[ARM_IDX]
    upper = kin._kin.upper_limits[ARM_IDX]
    tilt_tol = np.deg2rad(tilt_tol_deg)
    yaw_tol = np.deg2rad(yaw_tol_deg)

    bounds = [(float(lo), float(hi)) for lo, hi in zip(lower, upper)]

    result = minimize(
        _ik_cost,
        x0=q_arm_init.astype(float),
        args=(kin, q_head, T_target, pos_tol, tilt_tol, yaw_tol, z_weight),
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": max_iters, "ftol": 1e-12, "disp": verbose},
    )

    q_arm = result.x
    # Clamp to joint limits (SLSQP bounds should handle this, but be safe)
    q_arm = np.clip(q_arm, lower, upper)
    cost = float(result.fun)
    iters = int(result.nit) if hasattr(result, "nit") else result.nfev

    # Final FK to compute actual pose error
    T_final = kin.ee_in_base(q_head, q_arm)
    pos_err = float(np.linalg.norm(T_final[:3, 3] - T_target[:3, 3]))
    _, R_cur = T_final[:3, :3], T_target[:3, :3]

    return q_head.copy(), q_arm.copy(), pos_err, iters


def scipy_ik_multi_restart(
    kin,
    T_target: np.ndarray,
    q_head: np.ndarray,
    q_arm_current: np.ndarray,
    *,
    n_restarts: int = 5,
    noise_scale: float = 0.3,   # rad — roboarm 风格，稍大的噪声
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    多起点 SLSQP IK。

    Returns the best (lowest cost) solution.
    """
    best_result = None
    best_err = np.inf
    lower = kin._kin.lower_limits[ARM_IDX]
    upper = kin._kin.upper_limits[ARM_IDX]

    candidates = [q_arm_current.copy()]
    rng = np.random.RandomState(42)
    for _ in range(n_restarts - 1):
        noise = rng.randn(6) * noise_scale
        candidates.append(np.clip(q_arm_current + noise, lower, upper))

    for i, q0 in enumerate(candidates):
        try:
            q_h, q_a, err, it = scipy_ik(kin, T_target, q_head, q0, **kwargs)
            if err < best_err:
                best_err = err
                best_result = (q_h, q_a, err, it)
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("scipy_ik_multi_restart: all restarts failed")

    return best_result
