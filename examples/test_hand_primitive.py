#!/usr/bin/env python3
"""
Test the D1 Hand Primitive — exercise all 6 contracts.

This example verifies:

    1.  prm::hand.info            — query joint metadata
    2.  prm::hand.move.joint      — set joint positions
    3.  prm::hand.state.joint     — read joint positions
    4.  prm::hand.state.speed     — read joint speeds
    5.  prm::hand.set.speed       — set speed limits
    6.  prm::hand.set.torque      — set torque limits

Usage:
    python examples/test_hand_primitive.py
"""
from __future__ import annotations

import time
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clients.hand import D1HandPrimitive


def main():
    print("=" * 60)
    print("  D1 Hand Primitive — Contract Compliance Test")
    print("=" * 60)

    print("\033[91mWARNING: Keep the emergency stop within reach.\033[0m\n")

    # ── init ──────────────────────────────────────────────────────────────
    hand = D1HandPrimitive(hand_type="right", can_iface="can0")

    try:
        # ══════════════════════════════════════════════════════════════════
        # 1.  prm::hand.info  —  discover what we're talking to
        # ══════════════════════════════════════════════════════════════════
        print("─" * 40)
        print("[1] prm::hand.info")
        info = hand.info()
        print(f"    model      = {info['model']}")
        print(f"    num_joints = {info['num_joints']}")
        print(f"    joint_names = {info['joint_names']}")
        print(f"    hand_type  = {info['hand_type']}")
        n = info["num_joints"]

        # ══════════════════════════════════════════════════════════════════
        # 2.  prm::hand.state.joint  —  read current positions
        # ══════════════════════════════════════════════════════════════════
        print("\n─" * 40)
        print("[2] prm::hand.state.joint")
        pos = hand.state_joint()
        print(f"    initial positions = {[round(x, 3) for x in pos]}")

        # ══════════════════════════════════════════════════════════════════
        # 3.  prm::hand.set.speed + state.speed
        # ══════════════════════════════════════════════════════════════════
        print("\n─" * 40)
        print("[3] prm::hand.set.speed / prm::hand.state.speed")

        # Set moderate speed, then read back (returns instantaneous speed,
        # which is ~0 when stationary — this is correct hardware behaviour)
        hand.set_speed([0.3] * n)
        spd = hand.state_speed()
        assert len(spd) == n, f"expected {n} speeds, got {len(spd)}"
        print(f"    speed dim = {len(spd)}  (instantaneous speed: {[round(x, 3) for x in spd]})")

        # ══════════════════════════════════════════════════════════════════
        # 4.  prm::hand.set.torque
        # ══════════════════════════════════════════════════════════════════
        print("\n─" * 40)
        print("[4] prm::hand.set.torque")

        # Set moderate torque for reliable motion
        hand.set_torque([0.5] * n)
        print(f"    torque set to 0.5")

        # ══════════════════════════════════════════════════════════════════
        # 5.  prm::hand.move.joint  —  test several poses
        # ══════════════════════════════════════════════════════════════════
        print("\n─" * 40)
        print("[5] prm::hand.move.joint — pose sequence")

        # Pose A: half-close all fingers
        pose_a = [0.5] * n
        print(f"    → half-close: {pose_a}")
        hand.move_joint(pose_a)
        time.sleep(1.0)
        print(f"    actual: {[round(x, 3) for x in hand.state_joint()]}")

        # Pose B: pinch (thumb opposition + index close)
        pose_b = [0.5, 0.8, 0.8, 0.05, 0.05, 0.05]
        print(f"    → pinch:      {pose_b}")
        hand.move_joint(pose_b)
        time.sleep(1.0)
        print(f"    actual: {[round(x, 3) for x in hand.state_joint()]}")

        # Pose C: open all
        pose_c = [0.0] * n
        print(f"    → open:       {pose_c}")
        hand.move_joint(pose_c)
        time.sleep(1.0)
        print(f"    actual: {[round(x, 3) for x in hand.state_joint()]}")

        # ══════════════════════════════════════════════════════════════════
        # 6.  Error handling — wrong array length
        # ══════════════════════════════════════════════════════════════════
        print("\n─" * 40)
        print("[6] move.joint with wrong array length → expect ValueError")
        try:
            hand.move_joint([0.5, 0.5])  # only 2 values for a 6-joint hand
        except ValueError as e:
            print(f"    correctly rejected: {e}")

        # ══════════════════════════════════════════════════════════════════
        # Done
        # ══════════════════════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  All 6 contracts exercised successfully.")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        # ── safe shutdown ─────────────────────────────────────────────────
        print("\n[cleanup] opening hand and closing CAN …")
        hand.close()
        print("[cleanup] done.")


if __name__ == "__main__":
    main()
