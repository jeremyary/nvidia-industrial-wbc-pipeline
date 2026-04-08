# This project was developed with assistance from AI tools.
"""G1 29-DOF flat locomotion environment (Isaac Lab stock preset).

Same structure as the operator flat env but with Isaac Lab's stock default
positions, action scale (0.5), and no phase oscillator.
Obs dim remains 103 (phase term returns zeros for shape compatibility).
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from wbc_pipeline.envs.g1_29dof_shared_cfgs import (
    G1_29DOF_CommandsCfg,
    G1_29DOF_CurriculumCfg,
    G1_29DOF_EventsCfg,
    G1_29DOF_RewardsCfg,
    G1_29DOF_TerminationsCfg,
    apply_common_sim_settings,
    make_robot_cfg,
)
from wbc_pipeline.envs.joint_presets import ISAAC_LAB_PRESET
from wbc_pipeline.envs.mdp import phase_oscillator

# -- Active preset for this env module ----------------------------------------
_PRESET = ISAAC_LAB_PRESET

_ROBOT_29DOF = SceneEntityCfg("robot", joint_names=list(_PRESET.joint_order))


@configclass
class G1_29DOF_IsaacLabSceneCfg(InteractiveSceneCfg):
    """Scene with G1 29-DOF robot on flat terrain (Isaac Lab defaults)."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    robot = make_robot_cfg(_PRESET)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


@configclass
class G1_29DOF_IsaacLabObsCfg:
    """Observation groups — 103-dim (phase returns zeros when freq_hz=0)."""

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

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1_29DOF_IsaacLabActionsCfg:
    """29-dim joint position actions with Isaac Lab stock scale (0.5)."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_PRESET.joint_order),
        scale=_PRESET.action_scale,
        use_default_offset=True,
    )


@configclass
class G1_29DOF_IsaacLabFlatEnvCfg(ManagerBasedRLEnvCfg):
    """G1 29-DOF flat env with Isaac Lab stock defaults."""

    EXPECTED_OBS_DIM = 103
    EXPECTED_ACTION_DIM = 29

    scene: G1_29DOF_IsaacLabSceneCfg = G1_29DOF_IsaacLabSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: G1_29DOF_IsaacLabObsCfg = G1_29DOF_IsaacLabObsCfg()
    actions: G1_29DOF_IsaacLabActionsCfg = G1_29DOF_IsaacLabActionsCfg()

    commands: G1_29DOF_CommandsCfg = G1_29DOF_CommandsCfg()
    rewards: G1_29DOF_RewardsCfg = G1_29DOF_RewardsCfg()
    terminations: G1_29DOF_TerminationsCfg = G1_29DOF_TerminationsCfg()
    events: G1_29DOF_EventsCfg = G1_29DOF_EventsCfg()
    curriculum: G1_29DOF_CurriculumCfg = G1_29DOF_CurriculumCfg()

    def __post_init__(self):
        apply_common_sim_settings(self)
