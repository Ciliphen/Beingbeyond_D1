#!/usr/bin/env python3
"""
Convert D1 Zarr dataset to LeRobot v2.1 format for Being-H05 training.

Input:  Zarr dataset at ~/datasets/d1_teleop
Output: LeRobot-format dataset ready for Being-H05 post-training

Data mapping:
  Zarr                         → LeRobot parquet column
  ───────────────────────────────────────────────────────
  right_arm_qpos (6) +         → observation.state (12D)
  right_hand_qpos (6)

  action (12D)                 → action (12D)

  camera_0.rgb                 → videos/chunk-xxx/front_view/episode_xxxxxx.mp4
  camera_0.depth               → (discarded, Being-H uses RGB only)

  episode_ends                 → episodes.jsonl + episode boundaries
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import zarr


def convert_dataset(
    zarr_path: str,
    output_dir: str,
    task_description: str = "Demonstrated task",
    fps: int = 30,
):
    """Convert Zarr D1 dataset to LeRobot format."""

    zarr_path = Path(zarr_path)
    output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # 1. Load Zarr data
    # ------------------------------------------------------------------
    print(f"Loading Zarr from: {zarr_path}")
    root = zarr.open_group(str(zarr_path), mode="r")
    data = root["data"]
    meta = root["meta"]

    arr_action = np.array(data["action"])          # (T, 12)
    arr_arm = np.array(data["right_arm_qpos"])     # (T, 6)
    arr_hand = np.array(data["right_hand_qpos"])   # (T, 6)
    arr_rgb = np.array(data["camera_0.rgb"])       # (T, 256, 256, 3)

    # observation.state = arm + hand (12D)
    arr_state = np.concatenate([arr_arm, arr_hand], axis=1)  # (T, 12)

    episode_ends = np.array(meta["episode_ends"])             # indices, 1-based exclusive
    total_frames = arr_action.shape[0]
    total_episodes = len(episode_ends)

    action_dim = arr_action.shape[1]
    state_dim = arr_state.shape[1]

    print(f"  Frames: {total_frames}")
    print(f"  Episodes: {total_episodes}")
    print(f"  Actions: {action_dim}D, States: {state_dim}D")
    print(f"  Camera: {arr_rgb.shape[1]}×{arr_rgb.shape[2]}")
    print(f"  Total duration: ~{total_frames / fps / 60:.1f} min")

    # ------------------------------------------------------------------
    # 2. Create LeRobot directory structure
    # ------------------------------------------------------------------
    data_dir = output_dir / "data" / "chunk-000"
    video_dir = output_dir / "videos" / "chunk-000" / "front_view"
    meta_dir = output_dir / "meta"
    for d in [data_dir, video_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Slice episodes and write parquet + video
    # ------------------------------------------------------------------
    episodes_meta = []
    global_idx = 0
    prev_end = 0
    chunk_idx = 0

    for ep_idx in range(total_episodes):
        end_frame = int(episode_ends[ep_idx]) + 1  # episode_ends is exclusive final idx
        start_frame = prev_end
        length = end_frame - start_frame

        ep_name = f"episode_{ep_idx:06d}"
        print(f"  {ep_name}: frames {start_frame}→{end_frame-1} ({length} frames)")

        # ---- Parquet ----
        rows = []
        for local_idx in range(length):
            t = start_frame + local_idx
            rows.append({
                "timestamp": np.float32(local_idx / fps),
                "episode_index": np.int64(ep_idx),
                "frame_index": np.int64(local_idx),
                "index": np.int64(global_idx + local_idx),
                "task_index": np.int64(0),
                "observation.state": arr_state[t].astype(np.float32),
                "action": arr_action[t].astype(np.float32),
            })

        df = pd.DataFrame(rows)
        parquet_path = data_dir / f"{ep_name}.parquet"
        df.to_parquet(parquet_path, index=False)

        # ---- Video (MP4) ----
        video_path = video_dir / f"{ep_name}.mp4"
        # OpenCV VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = arr_rgb.shape[1], arr_rgb.shape[2]
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

        for local_idx in range(length):
            t = start_frame + local_idx
            frame_rgb = arr_rgb[t]                  # (H, W, 3), uint8
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        writer.release()

        # ---- Metadata ----
        episodes_meta.append({
            "episode_index": ep_idx,
            "length": length,
            "tasks": [task_description],
        })

        global_idx += length
        prev_end = end_frame

    # ------------------------------------------------------------------
    # 4. Write metadata files
    # ------------------------------------------------------------------
    print(f"\nWriting metadata...")

    # info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "beingbeyond_d1",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": [
                    "arm_joint_0", "arm_joint_1", "arm_joint_2",
                    "arm_joint_3", "arm_joint_4", "arm_joint_5",
                    "hand_joint_0", "hand_joint_1", "hand_joint_2",
                    "hand_joint_3", "hand_joint_4", "hand_joint_5",
                ],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": [
                    "arm_joint_0", "arm_joint_1", "arm_joint_2",
                    "arm_joint_3", "arm_joint_4", "arm_joint_5",
                    "hand_joint_0", "hand_joint_1", "hand_joint_2",
                    "hand_joint_3", "hand_joint_4", "hand_joint_5",
                ],
            },
            "observation.images.front_view": {
                "dtype": "video",
                "shape": [int(arr_rgb.shape[1]), int(arr_rgb.shape[2]), 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": fps,
                    "video.codec": "h264",
                    "video.height": int(arr_rgb.shape[1]),
                    "video.width": int(arr_rgb.shape[2]),
                    "video.channels": 3,
                    "video.is_depth_map": False,
                },
            },
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    # tasks.jsonl
    tasks = [{"task_index": 0, "description": task_description}]
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # stats.json
    print("Computing statistics...")
    stats = {}
    for col_name, col_data in [("action", arr_action), ("observation.state", arr_state)]:
        stats[col_name] = {
            "mean": col_data.mean(axis=0).tolist(),
            "std": col_data.std(axis=0).tolist(),
            "min": col_data.min(axis=0).tolist(),
            "max": col_data.max(axis=0).tolist(),
            "q01": np.percentile(col_data, 1, axis=0).tolist(),
            "q99": np.percentile(col_data, 99, axis=0).tolist(),
        }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone! Dataset saved to: {output_dir}")
    print(f"  Episodes: {total_episodes}")
    print(f"  Frames: {total_frames}")
    print(f"  Duration: ~{total_frames/fps/60:.1f} min")


def main():
    parser = argparse.ArgumentParser(
        description="Convert D1 Zarr dataset to LeRobot format"
    )
    parser.add_argument("--input", default="/home/xlf/datasets/d1_teleop",
                        help="Path to Zarr dataset directory")
    parser.add_argument("--output", required=True,
                        help="Output directory for LeRobot dataset")
    parser.add_argument("--task", default="Demonstrated task",
                        help="Task description for this dataset")
    parser.add_argument("--fps", type=int, default=30,
                        help="Recording FPS (default: 30)")
    args = parser.parse_args()

    convert_dataset(
        zarr_path=args.input,
        output_dir=args.output,
        task_description=args.task,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
