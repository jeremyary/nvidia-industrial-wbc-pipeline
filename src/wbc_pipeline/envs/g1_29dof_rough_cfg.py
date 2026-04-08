# This project was developed with assistance from AI tools.
"""G1 29-DOF rough terrain locomotion environment (operator preset).

Extends the flat env with procedurally generated rough terrain, a height
scanner (RayCasterCfg on torso_link), and terrain curriculum.

Obs dim: 103 (base) + 187 (height scan) = 290.
Action dim: 29 (same as flat).
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import rewards as vel_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums import terrain_levels_vel

from wbc_pipeline.envs.g1_29dof_shared_cfgs import (
    G1_29DOF_ActionsCfg,
    apply_common_sim_settings,
    make_robot_cfg,
)
from wbc_pipeline.envs.joint_presets import OPERATOR_PRESET
from wbc_pipeline.envs.mdp import height_scan, phase_oscillator

# -- Active preset ------------------------------------------------------------
_PRESET = OPERATOR_PRESET

_ROBOT_29DOF = SceneEntityCfg("robot", joint_names=list(_PRESET.joint_order))
_FEET_CFG = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])


# ==============================================================================
# Scene — rough terrain + height scanner
# ==============================================================================


@configclass
class G1_29DOF_RoughSceneCfg(InteractiveSceneCfg):
    """Scene with G1 29-DOF robot on procedurally generated rough terrain."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot = make_robot_cfg(_PRESET)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


# ==============================================================================
# Observations — 103 base + 187 height scan = 290
# ==============================================================================


@configclass
class G1_29DOF_RoughObsCfg:
    """Observation groups — 290-dim policy observations (103 base + 187 height scan)."""

    @configclass
    class PolicyCfg(ObsGroup):
        concatenate_terms = True

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _ROBOT_29DOF},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _ROBOT_29DOF},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)
        phase = ObsTerm(
            func=phase_oscillator,
            params={"freq_hz": _PRESET.phase_freq_hz, "offset": _PRESET.phase_offset},
        )
        height_scan_obs = ObsTerm(
            func=height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()


# ==============================================================================
# Rewards — adjusted for rough terrain
# ==============================================================================


@configclass
class G1_29DOF_RoughRewardsCfg:
    """Reward terms for rough terrain locomotion."""

    # Velocity tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"std": 0.25, "command_name": "base_velocity"}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"std": 0.25, "command_name": "base_velocity"}
    )

    # Penalties — relaxed for rough terrain
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)  # allow vertical motion on uneven ground
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # Feet
    feet_air_time = RewTerm(
        func=vel_rewards.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "command_name": "base_velocity",
            "threshold": 0.4,
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

    # Joint deviation penalties
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


# ==============================================================================
# Curriculum, Events, Terminations, Commands
# ==============================================================================


@configclass
class G1_29DOF_RoughCurriculumCfg:
    """Terrain difficulty curriculum."""

    terrain_levels = CurrTerm(func=terrain_levels_vel)


@configclass
class G1_29DOF_RoughEventsCfg:
    """Randomization events for rough terrain (includes external forces)."""

    reset_base = EventCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
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
    base_external_force_torque = EventCfg(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "force_range": (-5.0, 5.0),
            "torque_range": (-5.0, 5.0),
        },
    )


@configclass
class G1_29DOF_RoughCommandsCfg:
    """Velocity commands — forward-only bias for rough terrain training."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=False,
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class G1_29DOF_RoughTerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link"]), "threshold": 1.0},
    )


# ==============================================================================
# Top-level env config
# ==============================================================================


@configclass
class G1_29DOF_RoughEnvCfg(ManagerBasedRLEnvCfg):
    """G1 29-DOF rough terrain env producing 290-dim obs / 29-dim actions."""

    EXPECTED_OBS_DIM = 290
    EXPECTED_ACTION_DIM = 29

    scene: G1_29DOF_RoughSceneCfg = G1_29DOF_RoughSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: G1_29DOF_RoughObsCfg = G1_29DOF_RoughObsCfg()
    actions: G1_29DOF_ActionsCfg = G1_29DOF_ActionsCfg()
    commands: G1_29DOF_RoughCommandsCfg = G1_29DOF_RoughCommandsCfg()
    rewards: G1_29DOF_RoughRewardsCfg = G1_29DOF_RoughRewardsCfg()
    terminations: G1_29DOF_RoughTerminationsCfg = G1_29DOF_RoughTerminationsCfg()
    events: G1_29DOF_RoughEventsCfg = G1_29DOF_RoughEventsCfg()
    curriculum: G1_29DOF_RoughCurriculumCfg = G1_29DOF_RoughCurriculumCfg()

    def __post_init__(self):
        apply_common_sim_settings(self)
