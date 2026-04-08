# This project was developed with assistance from AI tools.
"""Training entrypoint with MLflow tracking, S3 checkpointing, and SIGTERM handling."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train G1 29-DOF locomotion policy with RSL-RL.")
parser.add_argument("--task", type=str, default="WBC-Velocity-Flat-G1-29DOF-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint or 's3' to resume latest from S3")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

import wbc_pipeline.envs  # noqa: E402, F401
from wbc_pipeline.config import TrainingConfig  # noqa: E402
from wbc_pipeline.runner import WBCRunner  # noqa: E402


def _resolve_entry_point(entry_point):
    """Resolve a string entry point to a config instance."""
    if isinstance(entry_point, str):
        import importlib

        mod_path, cls_name = entry_point.rsplit(":", 1)
        return getattr(importlib.import_module(mod_path), cls_name)()
    return entry_point()


def main():
    spec = gym.spec(args.task)
    env_cfg = _resolve_entry_point(spec.kwargs["env_cfg_entry_point"])
    agent_cfg: RslRlOnPolicyRunnerCfg = _resolve_entry_point(spec.kwargs["rsl_rl_cfg_entry_point"])

    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        env_cfg.seed = args.seed
        agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    training_cfg = TrainingConfig()
    log_dir = os.path.join("/workspace/isaaclab/logs/rsl_rl", agent_cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    runner = WBCRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, training_cfg=training_cfg)

    # Resume from checkpoint
    if args.resume == "s3":
        runner.resume_latest_from_s3()
    elif args.resume:
        runner.load(args.resume)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
