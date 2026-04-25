# This project was developed with assistance from AI tools.
"""Shared G1 29-DOF config components used across flat, rough, and warehouse variants.

Centralizes action, command, reward, termination, event, and curriculum configs
so that env variants import from here rather than cross-importing from each other.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import rewards as vel_rewards

from wbc_pipeline.envs.joint_presets import OPERATOR_PRESET, JointPreset

# -- Module-level preset (used by shared config classes) ----------------------
_PRESET = OPERATOR_PRESET
_FEET_CFG = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])


# -- Shared utility functions -------------------------------------------------


def make_robot_cfg(preset: JointPreset) -> ArticulationCfg:
    """Build G1 29-DOF robot config with the given preset's default positions."""
    cfg = G1_29DOF_CFG.copy()
    cfg.spawn.activate_contact_sensors = True
    cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos=dict(preset.default_positions),
        joint_vel={".*": 0.0},
    )
    return cfg


def apply_common_sim_settings(cfg) -> None:
    """Apply physics and simulation settings shared across all G1 29-DOF envs."""
    cfg.decimation = 4
    cfg.sim.dt = 0.005  # 200 Hz physics, 50 Hz control
    cfg.episode_length_s = 20.0
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physx.bounce_threshold_velocity = 0.2
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024


# -- Shared config classes ----------------------------------------------------


@configclass
class G1_29DOF_ActionsCfg:
    """29-dim joint position actions with operator preset's action scale."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_PRESET.joint_order),
        scale=_PRESET.action_scale,
        use_default_offset=True,
    )


@configclass
class G1_29DOF_CommandsCfg:
    """Velocity commands for locomotion."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=False,
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class G1_29DOF_RewardsCfg:
    """Reward terms adapted for 29-DOF G1 locomotion."""

    # Velocity tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"std": 0.25, "command_name": "base_velocity"}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"std": 0.25, "command_name": "base_velocity"}
    )

    # Penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # Feet
    feet_air_time = RewTerm(
        func=vel_rewards.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "command_name": "base_velocity",
            "threshold": 0.25,
        },
    )
    feet_slide = RewTerm(
        func=vel_rewards.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "asset_cfg": _FEET_CFG,
        },
    )

    # Joint deviation penalties (keep default pose)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"])},
    )


@configclass
class G1_29DOF_TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link"]), "threshold": 1.0},
    )


@configclass
class G1_29DOF_EventsCfg:
    """Randomization events."""

    reset_base = EventCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class G1_29DOF_CurriculumCfg:
    """No curriculum for flat env."""

    pass
