# This project was developed with assistance from AI tools.
"""Play a trained policy in Isaac Lab with GUI viewer or offscreen video recording."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize a trained G1 29-DOF policy.")
parser.add_argument("--task", type=str, default="WBC-Velocity-Flat-G1-29DOF-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_N.pt or s3://bucket/key")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=0, help="Max env steps (0 = unlimited)")
parser.add_argument("--video", action="store_true", default=False, help="Record offscreen video")
parser.add_argument("--video_length", type=int, default=500, help="Steps per video clip")
parser.add_argument("--video_dir", type=str, default="/tmp/isaaclab_videos", help="Video output directory")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402
import tempfile  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
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


def _download_s3_checkpoint(s3_uri: str) -> str:
    """Download checkpoint from s3://bucket/key and return local path."""
    import boto3

    path = s3_uri[len("s3://") :]
    bucket, key = path.split("/", 1)
    local_path = os.path.join(tempfile.mkdtemp(), os.path.basename(key))
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded {s3_uri} -> {local_path}")
    return local_path


def main():
    spec = gym.spec(args.task)
    env_cfg = _resolve_entry_point(spec.kwargs["env_cfg_entry_point"])
    agent_cfg: RslRlOnPolicyRunnerCfg = _resolve_entry_point(spec.kwargs["rsl_rl_cfg_entry_point"])

    env_cfg.scene.num_envs = args.num_envs

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)

    # Add scene lighting and set camera angle for rendering
    light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0))
    light_cfg.func("/World/defaultLight", light_cfg)
    env.unwrapped.sim.set_camera_view(eye=(3.0, 3.0, 2.0), target=(0.0, 0.0, 0.75))

    if args.video:
        os.makedirs(args.video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
        )
        if args.max_steps == 0:
            args.max_steps = args.video_length
    env = RslRlVecEnvWrapper(env)

    log_dir = os.path.join("/tmp/isaaclab_play", agent_cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    checkpoint = args.checkpoint
    if checkpoint.startswith("s3://"):
        checkpoint = _download_s3_checkpoint(checkpoint)
    runner.load(checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    step = 0
    with torch.inference_mode():
        while simulation_app.is_running() and (args.max_steps == 0 or step < args.max_steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            step += 1

    print(f"Finished {step} steps.")
    if args.video:
        print(f"Video saved to: {args.video_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
