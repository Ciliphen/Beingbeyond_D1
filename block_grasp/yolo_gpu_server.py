#!/usr/bin/env python3
"""
GPU-side YOLO inference server — runs in *bb_gpu* environment.

Reads pickled (frame, conf, iou) from stdin, writes pickled detections to stdout.
Exit when stdin is closed or receives an empty payload.

Usage (internal — launched by test_detect.py):
    conda activate bb_gpu
    python block_grasp/yolo_gpu_server.py --model path/to/best.pt
"""
from __future__ import annotations

import argparse
import io
import os
import pickle
import struct
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from object_detect import detect_objects_in_frame, load_model


def _recv_frame() -> tuple | None:
    """Read a length-prefixed pickled payload from stdin. Returns None on EOF."""
    # 4-byte length prefix (big-endian)
    raw_len = sys.stdin.buffer.read(4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    data = sys.stdin.buffer.read(msg_len)
    return pickle.loads(data)


def _send(obj) -> None:
    """Write a length-prefixed pickled payload to stdout."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.buffer.write(struct.pack(">I", len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Load model once
    model = load_model(args.model, device=args.device)
    print(f"[GPU Server] Model loaded on {args.device}, classes={list(model.names.values())}",
          file=sys.stderr, flush=True)

    while True:
        payload = _recv_frame()
        if payload is None:
            break

        # payload = (frame_bgr_bytes, conf_thres, iou_thres, infer_resolution)
        img_bytes, conf_thres, iou_thres, infer_size = payload
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Optional resize for speed
        if infer_size is not None:
            frame = cv2.resize(frame, infer_size, interpolation=cv2.INTER_AREA)

        t0 = time.time()
        detections = detect_objects_in_frame(model, frame, conf_thres=conf_thres, iou_thres=iou_thres)
        dt = time.time() - t0

        # Scale back to original resolution
        if infer_size is not None:
            orig_w, orig_h = infer_size  # these were the inf size
            sx = orig_w / frame.shape[1]  # no-op since we already resized
            sy = orig_h / frame.shape[0]
            detections = [
                ((u * sx, v * sy, w * sx, h * sy, r), s, c, n)
                for (u, v, w, h, r), s, c, n in detections
            ]

        _send({"detections": detections, "dt": dt})

    print("[GPU Server] Exiting.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
