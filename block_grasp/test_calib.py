#!/usr/bin/env python3
"""Quick test: click on image → see computed world coordinates."""
import math, os, sys
import cv2, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from camera.d1_camera_primitive import D1CameraPrimitive

CALIB = os.path.join(os.path.dirname(__file__), "handeye_calib.npz")
data = np.load(CALIB)
H = data["H"]
print(f"Loaded calib: H=\n{H}")
print(f"Head: yaw={math.degrees(float(data['head_yaw'])):.0f}°  pitch={math.degrees(float(data['head_pitch'])):.0f}°\n")

click = {"uv": None}

def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["uv"] = (x, y)

cam = D1CameraPrimitive(width=1280, height=720, fps=30)
cv2.namedWindow("Test Calib", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Test Calib", _on_mouse, click)

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

cam.close()
cv2.destroyAllWindows()
