# This project was developed with assistance from AI tools.
"""RSL-RL PPO runner config for G1 29-DOF locomotion."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1_29DOF_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config with [512, 256, 128] network matching the operator's architecture."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_29dof_flat"
    run_name = ""
    logger = "tensorboard"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1_29DOF_RoughPPORunnerCfg(G1_29DOF_PPORunnerCfg):
    """PPO config for rough terrain training."""

    experiment_name = "g1_29dof_rough"


@configclass
class G1_29DOF_WarehousePPORunnerCfg(G1_29DOF_PPORunnerCfg):
    """PPO config for warehouse scene training."""

    experiment_name = "g1_29dof_warehouse"


@configclass
class G1_29DOF_IsaacLabFlatPPORunnerCfg(G1_29DOF_PPORunnerCfg):
    """PPO config for Isaac Lab stock preset flat training."""

    experiment_name = "g1_29dof_isaaclab_flat"
