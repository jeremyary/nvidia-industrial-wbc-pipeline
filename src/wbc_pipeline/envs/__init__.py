# This project was developed with assistance from AI tools.
"""G1 29-DOF locomotion environments for ONNX-compatible policy training."""

try:
    import gymnasium as gym

    gym.register(
        id="WBC-Velocity-Flat-G1-29DOF-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "wbc_pipeline.envs.g1_29dof_env_cfg:G1_29DOF_FlatEnvCfg",
            "rsl_rl_cfg_entry_point": "wbc_pipeline.envs.g1_29dof_agent_cfg:G1_29DOF_PPORunnerCfg",
        },
    )

    gym.register(
        id="WBC-Velocity-Rough-G1-29DOF-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "wbc_pipeline.envs.g1_29dof_rough_cfg:G1_29DOF_RoughEnvCfg",
            "rsl_rl_cfg_entry_point": "wbc_pipeline.envs.g1_29dof_agent_cfg:G1_29DOF_RoughPPORunnerCfg",
        },
    )

    gym.register(
        id="WBC-Velocity-Warehouse-G1-29DOF-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "wbc_pipeline.envs.g1_29dof_scene_cfgs:G1_29DOF_WarehouseEnvCfg",
            "rsl_rl_cfg_entry_point": "wbc_pipeline.envs.g1_29dof_agent_cfg:G1_29DOF_WarehousePPORunnerCfg",
        },
    )

    gym.register(
        id="WBC-Velocity-IsaacLab-Flat-G1-29DOF-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "wbc_pipeline.envs.g1_29dof_isaaclab_flat_cfg:G1_29DOF_IsaacLabFlatEnvCfg",
            "rsl_rl_cfg_entry_point": "wbc_pipeline.envs.g1_29dof_agent_cfg:G1_29DOF_IsaacLabFlatPPORunnerCfg",
        },
    )
except ImportError:
    pass  # gymnasium not available outside the Isaac Lab container
