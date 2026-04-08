# This project was developed with assistance from AI tools.
"""Export a trained RSL-RL checkpoint to ONNX with observation normalization baked in."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export G1 29-DOF policy to ONNX.")
parser.add_argument("--task", type=str, default="WBC-Velocity-Flat-G1-29DOF-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_N.pt checkpoint")
parser.add_argument("--output_dir", type=str, default=".", help="Directory for exported ONNX file")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.exporter import export_policy_as_onnx  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import wbc_pipeline.envs  # noqa: E402, F401


def main():
    env_cfg_cls = gym.spec(args.task).kwargs["env_cfg_entry_point"]
    agent_cfg_cls = gym.spec(args.task).kwargs["rsl_rl_cfg_entry_point"]

    if isinstance(env_cfg_cls, str):
        import importlib

        mod_path, cls_name = env_cfg_cls.rsplit(":", 1)
        env_cfg = getattr(importlib.import_module(mod_path), cls_name)()
    else:
        env_cfg = env_cfg_cls()

    if isinstance(agent_cfg_cls, str):
        import importlib

        mod_path, cls_name = agent_cfg_cls.rsplit(":", 1)
        agent_cfg: RslRlOnPolicyRunnerCfg = getattr(importlib.import_module(mod_path), cls_name)()
    else:
        agent_cfg = agent_cfg_cls()

    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Derive obs/action dims from the live environment
    obs_dim = env.observation_space["policy"].shape[1]
    action_dim = env.num_actions

    # Validate against expected dims if the env cfg declares them
    expected_obs = getattr(type(env_cfg), "EXPECTED_OBS_DIM", None)
    expected_act = getattr(type(env_cfg), "EXPECTED_ACTION_DIM", None)
    if expected_obs is not None:
        assert obs_dim == expected_obs, f"Env obs dim {obs_dim} != expected {expected_obs}"
    if expected_act is not None:
        assert action_dim == expected_act, f"Env action dim {action_dim} != expected {expected_act}"
    print(f"Environment: obs_dim={obs_dim}, action_dim={action_dim}")

    log_dir = os.path.join("/workspace/isaaclab/logs/rsl_rl", agent_cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(args.checkpoint)

    policy = runner.alg.policy
    normalizer = policy.actor_obs_normalizer

    os.makedirs(args.output_dir, exist_ok=True)
    export_policy_as_onnx(policy, path=args.output_dir, normalizer=normalizer, filename="policy.onnx")

    onnx_path = os.path.join(args.output_dir, "policy.onnx")
    print(f"Exported ONNX to: {onnx_path}")

    # Validate exported model
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    input_name = model.graph.input[0].name
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    print(f"Input:  name={input_name!r}, shape={input_shape}")
    print(f"Output: name={model.graph.output[0].name!r}, shape={output_shape}")

    assert input_shape == [1, obs_dim], f"Expected input shape [1, {obs_dim}], got {input_shape}"
    assert output_shape == [1, action_dim], f"Expected output shape [1, {action_dim}], got {output_shape}"

    # Run inference with onnxruntime to verify outputs are valid
    session = ort.InferenceSession(onnx_path)
    dummy_obs = torch.randn(1, obs_dim).numpy()
    outputs = session.run(None, {input_name: dummy_obs})
    actions = outputs[0]
    print(f"ORT inference: input {dummy_obs.shape} -> output {actions.shape}")
    assert actions.shape == (1, action_dim), f"Expected ORT output (1, {action_dim}), got {actions.shape}"
    assert all(torch.isfinite(torch.tensor(actions)).flatten()), "ONNX output contains non-finite values"

    print("ONNX validation passed.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
