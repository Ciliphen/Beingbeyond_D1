# SPDX-License-Identifier: MulanPSL-2.0
"""Primitive-backed hardware adapters for the block-grasp skill.

The grasp pipeline (``block_grasp/grasp_controller.py``) is written against
three SDK-shaped handles — an arm ``robot``, a dexterous ``hand`` and a
``camera``. Instead of opening the serial link / CAN bus / RealSense directly,
this module exposes look-alike adapters that route every hardware call through
the robonix d1_arm / d1_hand / d1_camera *primitives* over gRPC. The skill thus
becomes a pure consumer; the primitives own the hardware.

Mapping (only the methods grasp_controller actually calls):

  robot.get_positions()        -> arm/get_state        (+ cached head pose)
  robot.set_positions(q)       -> arm/set_head (on head change) + arm/move_joint
  robot.wait_until_reached(q)  -> poll arm/get_state until arm joints converge
  robot.close()                -> no-op (primitive owns the serial link)

  hand.set_joint_pos(vec)      -> hand/move_joint (named JointValue[])
  hand.read_joint_pos()        -> hand/get_state
  hand.open_hand()             -> hand/move_joint all-open (safe release)
  hand.close_can()             -> no-op (primitive owns the CAN bus)

  camera.get_aligned_frames()  -> camera/snapshot (JPEG Image -> RGB ndarray)
  camera.stop()                -> no-op (primitive owns the RealSense)

Arm joint I/O is joint-space (the skill keeps its own IK/FK): the 6 arm joints
go through move_joint and the 2 head joints through set_head, so no Cartesian
move_pose is used. get_state/move_joint expose ONLY the 6 arm joints; the head
is held constant after set_head, so get_positions() prepends the last commanded
head pose to reconstruct the 8-vector grasp_controller expects.
"""
from __future__ import annotations

import time

import numpy as np


class PrimitiveArm:
    """HeadArmRobot-look-alike backed by the arm primitive (joint-space)."""

    def __init__(
        self,
        *,
        get_state_stub,
        move_joint_stub,
        set_head_stub,
        arm_pb2,
        arm_tol: float = 0.03,
        head_eps: float = 1e-4,
        timeout: float = 8.0,
        poll: float = 0.03,
        settle_eps: float = 0.003,
        settle_polls: int = 3,
    ) -> None:
        self._gs = get_state_stub
        self._mj = move_joint_stub
        self._sh = set_head_stub
        self._pb = arm_pb2
        self._arm_tol = arm_tol
        self._head_eps = head_eps
        self._timeout = timeout
        self._poll = poll
        # "Settled" = the arm stopped moving (steady-state offset from target):
        # < settle_eps rad of change across settle_polls consecutive reads.
        self._settle_eps = settle_eps
        self._settle_polls = settle_polls
        # Last head pose we commanded (yaw, pitch) rad; None until first set_head.
        self._head: tuple[float, float] | None = None

    # -- reads ---------------------------------------------------------------
    def _read_arm(self) -> list[float]:
        resp = self._gs.GetJointState(self._pb.GetJointState_Request())
        if not resp.ok:
            raise RuntimeError(f"arm get_state failed: {resp.message}")
        return [float(v) for v in resp.positions]

    def get_positions(self) -> np.ndarray:
        arm6 = self._read_arm()
        yaw, pitch = self._head if self._head is not None else (0.0, 0.0)
        return np.asarray([yaw, pitch, *arm6], dtype=float)

    def get_positions_and_velocities(self):
        resp = self._gs.GetJointState(self._pb.GetJointState_Request())
        if not resp.ok:
            raise RuntimeError(f"arm get_state failed: {resp.message}")
        yaw, pitch = self._head if self._head is not None else (0.0, 0.0)
        pos = np.asarray([yaw, pitch, *[float(v) for v in resp.positions]], dtype=float)
        vel = np.asarray([0.0, 0.0, *[float(v) for v in resp.velocities]], dtype=float)
        return pos, vel

    # -- writes --------------------------------------------------------------
    def _set_head(self, yaw: float, pitch: float) -> None:
        resp = self._sh.SetHead(self._pb.SetHead_Request(yaw=float(yaw), pitch=float(pitch)))
        if not resp.ok:
            raise RuntimeError(f"arm set_head failed: {resp.message}")
        self._head = (float(yaw), float(pitch))

    @staticmethod
    def _split(q) -> tuple[tuple[float, float] | None, list[float]]:
        q = [float(v) for v in q]
        if len(q) == 8:
            return (q[0], q[1]), q[2:8]
        if len(q) == 6:
            return None, q
        raise ValueError(f"expected 6 or 8 joint values, got {len(q)}")

    def set_positions(self, q) -> None:
        head, arm6 = self._split(q)
        if head is not None and (
            self._head is None
            or abs(head[0] - self._head[0]) > self._head_eps
            or abs(head[1] - self._head[1]) > self._head_eps
        ):
            self._set_head(head[0], head[1])
        resp = self._mj.MoveJoint(self._pb.MoveJoint_Request(positions=arm6))
        if not resp.ok:
            raise RuntimeError(f"arm move_joint failed: {resp.message}")

    def wait_until_reached(self, q, active_joint_indices=None, timeout: float | None = None) -> bool:
        """Block until the 6 arm joints reach the target OR the arm has moved and
        then stopped (settled). The controller commands IK joint targets the
        hardware tracks with a small steady-state offset, so an exact target match
        may never converge and would stall until timeout — hence the settle check.

        CRITICAL: settle is only accepted AFTER motion has actually begun. Right
        after a command is dispatched the arm has not started moving yet, so
        consecutive reads are identical; without the ``moved`` latch that would be
        misread as "already settled" and this would return before the arm moves at
        all — e.g. the startup lift would report done while the arm is still low,
        and the following wrist rotation would swing into the table.

        Head is positioned by the blocking set_head, so it is not polled here.
        Returns True on reached/settled, False on timeout (caller proceeds)."""
        _ = active_joint_indices
        _head, arm6 = self._split(q)
        deadline = time.time() + (self._timeout if timeout is None else timeout)
        prev: list[float] | None = None
        still = 0
        moved = False
        while time.time() < deadline:
            cur = self._read_arm()
            # reached: close to the commanded target → done
            if max((abs(c - t) for c, t in zip(cur, arm6)), default=0.0) <= self._arm_tol:
                return True
            if prev is not None:
                delta = max((abs(c - p) for c, p in zip(cur, prev)), default=0.0)
                if delta > self._settle_eps:
                    moved = True   # motion has begun
                    still = 0
                elif moved:        # settled, but only once the arm actually moved
                    still += 1
                    if still >= self._settle_polls:
                        return True
            prev = cur
            time.sleep(self._poll)
        return False

    def close(self) -> None:
        pass  # the arm primitive owns the serial link


class PrimitiveHand:
    """DexHand-look-alike backed by the hand primitive (normalized [0,1])."""

    def __init__(self, *, move_joint_stub, get_state_stub, hand_pb2, axis_names) -> None:
        self._mj = move_joint_stub
        self._gs = get_state_stub
        self._pb = hand_pb2
        self._names = list(axis_names)

    def set_joint_pos(self, vec) -> None:
        vec = [float(v) for v in vec]
        if len(vec) != len(self._names):
            raise ValueError(f"expected {len(self._names)} hand values, got {len(vec)}")
        targets = [self._pb.JointValue(name=n, value=v) for n, v in zip(self._names, vec)]
        resp = self._mj.MoveJoint(self._pb.MoveJoint_Request(targets=targets))
        if not resp.ok:
            raise RuntimeError(f"hand move_joint failed: {resp.message}")

    def read_joint_pos(self) -> list[float]:
        resp = self._gs.GetJointState(self._pb.GetJointState_Request())
        if not resp.ok:
            raise RuntimeError(f"hand get_state failed: {resp.message}")
        return [float(v) for v in resp.positions]

    def open_hand(self) -> None:
        # Safe release on shutdown: fully open (all axes 0.0).
        self.set_joint_pos([0.0] * len(self._names))

    def close_can(self) -> None:
        pass  # the hand primitive owns the CAN bus


class PrimitiveCamera:
    """RealSenseCamera-look-alike backed by the camera primitive (RGB only)."""

    def __init__(self, *, snapshot_stub, camera_pb2) -> None:
        self._ss = snapshot_stub
        self._pb = camera_pb2

    def get_aligned_frames(self, filtered: bool = False):
        """Return (rgb, None). The grasp pipeline uses 2D homography and ignores
        depth, so the camera primitive only serves RGB; depth is None."""
        _ = filtered
        resp = self._ss.GetCameraImage(self._pb.GetCameraImage_Request())
        data = bytes(resp.image.data)
        if not data:
            raise RuntimeError("camera snapshot returned no image (camera not initialized?)")
        from io import BytesIO

        from PIL import Image as PILImage

        rgb = np.asarray(PILImage.open(BytesIO(data)).convert("RGB"))
        return rgb, None

    def stop(self) -> None:
        pass  # the camera primitive owns the RealSense


def connect_primitives(skill):
    """Discover + connect the d1 arm/hand/camera primitives via atlas and return
    ``(arm, hand, camera)`` adapter handles. Raises RuntimeError if any primitive
    is not registered / reachable. Channels are tracked by ``skill`` and closed
    on teardown."""
    import grpc
    from robonix_api import ATLAS, Transport

    # Generated by rbnx codegen into rbnx-build/codegen/proto_gen (added to
    # sys.path by the Skill constructor before this module is imported).
    import arm_pb2  # type: ignore
    import camera_pb2  # type: ignore
    import hand_pb2  # type: ignore
    import robonix_contracts_pb2_grpc as cg  # type: ignore

    def _stub(contract: str, stub_cls):
        caps = ATLAS.find_capability(contract_id=contract, transport=Transport.GRPC)
        if not caps:
            raise RuntimeError(
                f"no gRPC provider for {contract} — is the primitive deployed and running?"
            )
        ch = skill.connect_capability(caps[0], contract, Transport.GRPC)
        return stub_cls(grpc.insecure_channel(ch.endpoint))

    arm = PrimitiveArm(
        get_state_stub=_stub("robonix/primitive/arm/get_state", cg.RobonixPrimitiveArmGetStateStub),
        move_joint_stub=_stub("robonix/primitive/arm/move_joint", cg.RobonixPrimitiveArmMoveJointStub),
        set_head_stub=_stub("robonix/primitive/arm/set_head", cg.RobonixPrimitiveArmSetHeadStub),
        arm_pb2=arm_pb2,
    )

    info = _stub("robonix/primitive/hand/info", cg.RobonixPrimitiveHandInfoStub).GetHandInfo(
        hand_pb2.GetHandInfo_Request()
    )
    if not info.ok:
        raise RuntimeError(f"hand info failed: {info.message}")
    hand = PrimitiveHand(
        move_joint_stub=_stub("robonix/primitive/hand/move_joint", cg.RobonixPrimitiveHandMoveJointStub),
        get_state_stub=_stub("robonix/primitive/hand/get_state", cg.RobonixPrimitiveHandGetStateStub),
        hand_pb2=hand_pb2,
        axis_names=[j.name for j in info.joints],
    )

    camera = PrimitiveCamera(
        snapshot_stub=_stub("robonix/primitive/camera/snapshot", cg.RobonixPrimitiveCameraSnapshotStub),
        camera_pb2=camera_pb2,
    )
    return arm, hand, camera
