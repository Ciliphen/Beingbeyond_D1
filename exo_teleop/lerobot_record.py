#!/usr/bin/env python3
"""
LeRobot-format data recorder for BeingBeyond D1 with exoskeleton teleoperation.

Records D1 demonstration episodes in LeRobot v2.1 format, directly compatible
with Being-H05 post-training.

Usage:
    python lerobot_record.py \
        --output /path/to/d1_data/pick_cube \
        --task "Pick the cube and place it on the plate" \
        --fps 30 \
        --episode-timeout 120

Controls during recording:
    Enter  → start a new episode (also saves the previous one)
    q      → quit recording
    Ctrl+C → safe shutdown

Output structure (LeRobot v2.1):
    {output}/
    ├── data/chunk-000/episode_000000.parquet
    ├── videos/chunk-000/front_view/episode_000000.mp4
    └── meta/
        ├── info.json
        ├── episodes.jsonl
        ├── tasks.jsonl
        └── stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from teleop_real import TeleopReal, TeleopRealCfg, ARM_INIT_RAD, deg_list_to_rad
from vision import RealSenseCamera
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
# D1 joint layout (matching teleop_real.py)
# ==============================================================================
JOINT_LAYOUT = {
    "head": ["head_pan", "head_tilt"],                         # 2 joints
    "arm":  ["joint_1", "joint_2", "joint_3",                  # 6 joints
             "joint_4", "joint_5", "joint_6"],
    "hand": ["thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_ip",    # 6 joints
             "index_mcp_pitch", "index_dip",
             "middle_mcp_pitch"],   # simplified name list (actual hand has more)
}
TOTAL_JOINTS = 14  # 2 (head) + 6 (arm) + 6 (hand)
FPS = 30


# ==============================================================================
# LeRobot Recorder
# ==============================================================================

class LeRobotRecorder:
    """Records D1 teleoperation data in LeRobot v2.1 format."""

    def __init__(
        self,
        output_dir: str,
        task_description: str,
        fps: int = 30,
        episode_timeout: float = 300.0,
        camera_width: int = 640,
        camera_height: int = 480,
    ):
        self.output_dir = Path(output_dir)
        self.task_description = task_description
        self.fps = fps
        self.episode_timeout = episode_timeout
        self.camera_width = camera_width
        self.camera_height = camera_height

        # Episode tracking
        self.episode_index = 0
        self.frame_index = 0
        self.global_index = 0
        self.episode_metadata: list[dict] = []

        # Initialize directory structure
        self.data_dir = self.output_dir / "data" / "chunk-000"
        self.video_dir = self.output_dir / "videos" / "chunk-000" / "front_view"
        self.meta_dir = self.output_dir / "meta"
        for d in [self.data_dir, self.video_dir, self.meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Task list
        self.tasks: list[dict] = []
        self._register_task()

        # Recording state
        self._recording = False
        self._running = True
        self._frames_buffer: list[dict] = []
        self._video_frames: list[np.ndarray] = []
        self._start_time = 0.0

        # Camera
        self._camera: RealSenseCamera | None = None

    def _register_task(self):
        """Register the task in tasks.jsonl (dedup by description)."""
        tasks_file = self.meta_dir / "tasks.jsonl"
        if tasks_file.exists():
            with open(tasks_file) as f:
                for line in f:
                    t = json.loads(line.strip())
                    self.tasks.append(t)

        # Check if this task already exists
        existing = [t for t in self.tasks if t["description"] == self.task_description]
        if existing:
            self.task_index = existing[0]["task_index"]
        else:
            self.task_index = len(self.tasks)
            self.tasks.append({
                "task_index": self.task_index,
                "description": self.task_description,
            })
            with open(tasks_file, "w") as f:
                for t in self.tasks:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # ---- Camera ----

    def _init_camera(self):
        self._camera = RealSenseCamera(
            width=self.camera_width,
            height=self.camera_height,
            hz=self.fps,
        )
        # Let camera auto-exposure settle
        time.sleep(1.0)

    def _get_rgb_frame(self) -> np.ndarray:
        """Get RGB frame from RealSense camera."""
        assert self._camera is not None
        color_np, _ = self._camera.get_aligned_frames()  # RGB, depth
        return color_np.copy()

    # ---- Recording ----

    def start_episode(self):
        """Begin recording a new episode."""
        self._recording = True
        self._frames_buffer = []
        self._video_frames = []
        self.frame_index = 0
        self._start_time = time.time()

        episode_name = f"episode_{self.episode_index:06d}"
        print(f"\n{'='*60}")
        print(f"▶ Recording {episode_name}")
        print(f"  Task: {self.task_description}")
        print(f"  Timeout: {self.episode_timeout}s")
        print(f"{'='*60}")

    def stop_episode(self):
        """Stop recording and save the episode."""
        if not self._recording:
            return

        self._recording = False
        episode_name = f"episode_{self.episode_index:06d}"
        duration = time.time() - self._start_time

        if len(self._frames_buffer) < 10:
            print(f"✗ {episode_name}: too short ({len(self._frames_buffer)} frames), discarding")
            self._frames_buffer = []
            self._video_frames = []
            return

        # Save parquet
        parquet_path = self.data_dir / f"{episode_name}.parquet"
        df = pd.DataFrame(self._frames_buffer)
        df.to_parquet(parquet_path, index=False)

        # Save video
        video_path = self.video_dir / f"{episode_name}.mp4"
        self._save_video(video_path)

        # Record episode metadata
        length = len(self._frames_buffer)
        self.episode_metadata.append({
            "episode_index": self.episode_index,
            "length": length,
            "tasks": [self.task_description],
        })
        self.global_index += length
        self.episode_index += 1

        print(f"✓ {episode_name} saved: {length} frames ({duration:.1f}s)")
        print(f"  Parquet: {parquet_path}")
        print(f"  Video:   {video_path}")
        self._frames_buffer = []
        self._video_frames = []

    def record_frame(self, q_target: np.ndarray, q_actual: np.ndarray):
        """Record one frame of data.

        Args:
            q_target: 14-dim commanded joint positions (from exo)
            q_actual: 14-dim actual robot joint positions (from robot)
        """
        if not self._recording:
            return

        # Check timeout
        if time.time() - self._start_time > self.episode_timeout:
            print(f"  ⚠ Episode timeout, auto-stopping")
            self.stop_episode()
            return

        # Capture frame
        try:
            rgb = self._get_rgb_frame()
        except Exception:
            # Camera might not be ready, skip this frame
            return

        # Build LeRobot-format row
        timestamp = time.time() - self._start_time
        row = {
            "timestamp": np.float32(timestamp),
            "episode_index": np.int64(self.episode_index),
            "frame_index": np.int64(self.frame_index),
            "index": np.int64(self.global_index + self.frame_index),
            "task_index": np.int64(self.task_index),
            # State: actual robot joint positions (14 dims, radians)
            "observation.state": q_actual.astype(np.float32),
            # Action: commanded target positions (14 dims, radians)
            "action": q_target.astype(np.float32),
        }

        self._frames_buffer.append(row)
        self._video_frames.append(rgb)
        self.frame_index += 1

    def _save_video(self, video_path: Path):
        """Save recorded frames as MP4 video using OpenCV."""
        if not self._video_frames:
            return

        h, w = self._video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (w, h))

        for frame in self._video_frames:
            # OpenCV expects BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        writer.release()

    # ---- Metadata ----

    def save_metadata(self):
        """Generate all LeRobot metadata files."""
        print(f"\nGenerating metadata...")

        # info.json
        total_frames = sum(ep["length"] for ep in self.episode_metadata)
        info = {
            "codebase_version": "v2.1",
            "robot_type": "beingbeyond_d1",
            "total_episodes": len(self.episode_metadata),
            "total_frames": total_frames,
            "total_tasks": len(self.tasks),
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self.episode_metadata)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "action": {
                    "dtype": "float32",
                    "shape": [TOTAL_JOINTS],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [TOTAL_JOINTS],
                },
                "observation.images.front_view": {
                    "dtype": "video",
                    "shape": [self.camera_height, self.camera_width, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.fps": self.fps,
                        "video.codec": "h264",
                        "video.height": self.camera_height,
                        "video.width": self.camera_width,
                        "video.channels": 3,
                        "video.is_depth_map": False,
                    },
                },
            },
        }

        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        # episodes.jsonl
        with open(self.meta_dir / "episodes.jsonl", "w") as f:
            for ep in self.episode_metadata:
                f.write(json.dumps(ep) + "\n")

        # stats.json (compute global statistics)
        self._compute_stats()

        print(f"  Episodes: {len(self.episode_metadata)}")
        print(f"  Total frames: {total_frames}")
        print(f"  Total minutes: {total_frames / self.fps / 60:.1f}")
        print(f"  Metadata saved to: {self.meta_dir}")

    def _compute_stats(self):
        """Compute global statistics across all episodes."""
        if not self.episode_metadata:
            return

        # Collect all actions and states
        all_actions = []
        all_states = []

        for ep_meta in self.episode_metadata:
            ep_idx = ep_meta["episode_index"]
            parquet_path = self.data_dir / f"episode_{ep_idx:06d}.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                actions = np.stack(df["action"].values)
                states = np.stack(df["observation.state"].values)
                all_actions.append(actions)
                all_states.append(states)

        if not all_actions:
            return

        all_actions = np.concatenate(all_actions, axis=0)  # (N, 14)
        all_states = np.concatenate(all_states, axis=0)

        # Compute stats per dimension
        def compute_stats(arr: np.ndarray, name: str) -> dict:
            return {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
                "q01": np.percentile(arr, 1, axis=0).tolist(),
                "q99": np.percentile(arr, 99, axis=0).tolist(),
            }

        stats = {
            "action": compute_stats(all_actions, "action"),
            "observation.state": compute_stats(all_states, "observation.state"),
        }

        with open(self.meta_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)


# ==============================================================================
# Main Recording Loop
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Record D1 teleoperation data in LeRobot format"
    )
    parser.add_argument("--output", required=True,
                        help="Output directory for LeRobot dataset")
    parser.add_argument("--task", default="Pick the cube and place it on the plate",
                        help="Task description")
    parser.add_argument("--fps", type=int, default=30,
                        help="Recording FPS (default: 30)")
    parser.add_argument("--episode-timeout", type=float, default=300.0,
                        help="Max episode duration in seconds (default: 300)")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--head-deg", type=float, nargs=2, default=[-30.0, 15.0],
                        help="Head pan/tilt in degrees (default: -30 15)")
    parser.add_argument("--arm-port", default="/dev/arm_exo")
    parser.add_argument("--hand-port", default="/dev/hand_exo")
    parser.add_argument("--robot-port", default="/dev/ttyUSB0")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   D1 LeRobot Data Recorder (Exoskeleton Teleop)         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nOutput:    {args.output}")
    print(f"Task:      {args.task}")
    print(f"FPS:       {args.fps}")
    print(f"\n\033[91mWARNING: Keep emergency stop within reach at all times!\033[0m")

    recorder = LeRobotRecorder(
        output_dir=args.output,
        task_description=args.task,
        fps=args.fps,
        episode_timeout=args.episode_timeout,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )

    # Initialize camera
    print("\nInitializing RealSense camera...")
    recorder._init_camera()
    print("Camera ready.")

    # Initialize teleop
    head_deg = tuple(args.head_deg)
    cfg = TeleopRealCfg(
        arm_port=args.arm_port,
        hand_port=args.hand_port,
        robot_arm_dev=args.robot_port,
        head_deg=head_deg,
        enable_vision=False,  # We handle camera ourselves
    )

    print("Initializing exoskeleton teleoperation...")
    teleop = TeleopReal(cfg)
    teleop._wait_ready()
    teleop._open_robot()
    teleop._move_to_initial_pose()

    print("\n" + "=" * 60)
    print("READY. Controls:")
    print("  [Enter]  → Start/Stop recording an episode")
    print("  [q]      → Quit")
    print("  [Ctrl+C] → Safe shutdown")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Recording loop
    # ------------------------------------------------------------------
    dt = 1.0 / args.fps
    head_rad = np.deg2rad(np.asarray(head_deg, dtype=np.float64))

    import select
    import termios
    import tty

    def kbhit():
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        return dr != []

    def getch():
        return sys.stdin.read(1)

    # Set terminal to raw mode for non-blocking input
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    try:
        while recorder._running:
            t0 = time.perf_counter()

            # ---- Handle keyboard ----
            if kbhit():
                ch = getch()
                if ch == "\n" or ch == "\r":
                    if recorder._recording:
                        recorder.stop_episode()
                    else:
                        recorder.start_episode()
                elif ch == "q":
                    if recorder._recording:
                        recorder.stop_episode()
                    recorder._running = False
                    break

            # ---- Teleop + Record ----
            # Read exo and build target
            teleop.teleop = teleop._build_teleop()  # 14D: head(2)+arm(6)+hand(6)

            # Apply to robot
            teleop._apply(teleop.teleop)

            # Read actual robot state
            q_actual = np.array(teleop.robot.get_q(), dtype=np.float64)  # 14D from robot

            # Record frame
            recorder.record_frame(
                q_target=teleop.teleop.copy(),
                q_actual=q_actual,
            )

            # Timing
            elapsed = time.perf_counter() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        # ---- Cleanup ----
        _restore_terminal(fd, old_settings)

        if recorder._recording:
            recorder.stop_episode()

        print("\nSaving metadata...")
        recorder.save_metadata()

        print("Safe shutdown...")
        teleop._safe_exit()
        teleop.close()

        if recorder._camera:
            recorder._camera.stop()

        print(f"\nDone! Dataset saved to: {args.output}")
        print(f"Total episodes: {recorder.episode_index}")


def _restore_terminal(fd, old_settings):
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except Exception:
        pass


if __name__ == "__main__":
    main()
