#!/usr/bin/env python3
"""Quick test: click on image → see computed world coordinates."""
import math, os, sys, time
import cv2, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from clients.camera import D1CameraPrimitive
from beingbeyond_d1_sdk.head_arm import HeadArmRobot
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path

CALIB = os.path.join(os.path.dirname(__file__), "handeye_calib.npz")
data = np.load(CALIB)
H = data["H"]
head_yaw = float(data["head_yaw"])
head_pitch = float(data["head_pitch"])
print(f"Loaded calib:\n  H=\n{H}")
print(f"  Head: yaw={math.degrees(head_yaw):.0f}°  pitch={math.degrees(head_pitch):.0f}°\n")

# Set head to calibration position
print("[Init] Robot + set head to calib position ...")
robot = HeadArmRobot(urdf_path=get_default_urdf_path(), dev="/dev/ttyUSB0", baudrate=1_000_000)
q_init = np.radians([0, 0, 0, -60, 60, 0, 0, 0])
robot.set_positions(q_init)
robot.wait_until_reached(q_init, active_joint_indices=range(8))
time.sleep(0.3)
q = np.asarray(robot.get_positions(), dtype=float)
q[0] = head_yaw
q[1] = head_pitch
robot.set_positions(q)
robot.wait_until_reached(q, active_joint_indices=[0, 1])
print(f"  Head set: yaw={math.degrees(q[0]):.0f}°  pitch={math.degrees(q[1]):.0f}°\n")

click = {"uv": None}
def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["uv"] = (x, y)

cam = D1CameraPrimitive(width=1280, height=720, fps=30)
cv2.namedWindow("Test Calib", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Test Calib", _on_mouse, click)

try:
    while True:
        rgb = cam.snapshot()
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if click["uv"] is not None:
            u, v = click["uv"]
            click["uv"] = None
            p = np.array([u, v, 1.0])
            w = H @ p; w /= w[2]
            print(f"pixel=({u},{v}) → world=({w[0]:.3f}, {w[1]:.3f})")

        cv2.imshow("Test Calib", vis)
        if cv2.waitKey(5) & 0xFF == 27:
            break
finally:
    cam.close()
    robot.close()
    cv2.destroyAllWindows()
