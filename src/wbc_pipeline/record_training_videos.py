# This project was developed with assistance from AI tools.
"""Record videos from multiple training checkpoints in a single Isaac Sim session."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batch-record training checkpoint videos.")
parser.add_argument("--task", type=str, default="WBC-Velocity-Flat-G1-29DOF-v0")
parser.add_argument("--run_name", type=str, default="flat", help="Name prefix for video files")
parser.add_argument("--checkpoint_dir", type=str, default="/checkpoints", help="Directory containing .pt files")
parser.add_argument("--video_dir", type=str, default="/videos", help="Output directory for videos")
parser.add_argument("--every_n", type=int, default=2500, help="Record every N iterations (0 = all)")
parser.add_argument("--steps", type=int, default=500, help="Steps per video")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402
import re  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import wbc_pipeline.envs  # noqa: E402, F401


def _resolve_entry_point(entry_point):
    if isinstance(entry_point, str):
        import importlib

        mod_path, cls_name = entry_point.rsplit(":", 1)
        return getattr(importlib.import_module(mod_path), cls_name)()
    return entry_point()


def _find_checkpoints(checkpoint_dir: str, every_n: int) -> list[tuple[int, str]]:
    """Find .pt checkpoint files and return sorted (iteration, path) pairs."""
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        match = re.match(r"model_(\d+)\.pt$", f)
        if match:
            iter_num = int(match.group(1))
            checkpoints.append((iter_num, os.path.join(checkpoint_dir, f)))

    checkpoints.sort(key=lambda x: x[0])

    if every_n > 0 and checkpoints:
        # Always include first, last, and every Nth
        last_iter = checkpoints[-1][0]
        filtered = []
        for iter_num, path in checkpoints:
            if iter_num % every_n == 0 or iter_num == last_iter:
                filtered.append((iter_num, path))
        if not filtered:
            filtered = [checkpoints[-1]]
        checkpoints = filtered

    return checkpoints


def main():
    os.makedirs(args.video_dir, exist_ok=True)

    spec = gym.spec(args.task)
    env_cfg = _resolve_entry_point(spec.kwargs["env_cfg_entry_point"])
    agent_cfg: RslRlOnPolicyRunnerCfg = _resolve_entry_point(spec.kwargs["rsl_rl_cfg_entry_point"])
    env_cfg.scene.num_envs = args.num_envs

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")

    # Add scene lighting and set camera
    light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0))
    light_cfg.func("/World/defaultLight", light_cfg)
    env.unwrapped.sim.set_camera_view(eye=(3.0, 3.0, 2.0), target=(0.0, 0.0, 0.75))

    env_wrapped = RslRlVecEnvWrapper(env)

    log_dir = os.path.join("/tmp/isaaclab_record", agent_cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    checkpoints = _find_checkpoints(args.checkpoint_dir, args.every_n)
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints to record")
    print(f"Iterations: {', '.join(str(n) for n, _ in checkpoints)}")
    print()

    for i, (iter_num, ckpt_path) in enumerate(checkpoints):
        video_name = f"{args.run_name}_iter_{iter_num:06d}.mp4"
        video_path = os.path.join(args.video_dir, video_name)

        if os.path.exists(video_path):
            print(f"[{i + 1}/{len(checkpoints)}] Skipping iter {iter_num} (video exists)")
            continue

        print(f"[{i + 1}/{len(checkpoints)}] Recording iter {iter_num}...")

        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=env_wrapped.unwrapped.device)

        # Reset and record
        obs = env_wrapped.get_observations()
        frames = []
        with torch.inference_mode():
            for step in range(args.steps):
                actions = policy(obs)
                obs, _, _, _ = env_wrapped.step(actions)
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))

        if frames:
            imageio.mimwrite(video_path, frames, fps=50, quality=8)
            print(f"  Saved {video_path} ({len(frames)} frames)")
        else:
            print(f"  WARNING: No frames captured for iter {iter_num}")

    print()
    print(f"Done. {len(checkpoints)} videos in {args.video_dir}/")


if __name__ == "__main__":
    main()
    simulation_app.close()
