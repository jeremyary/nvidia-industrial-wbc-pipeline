# This project was developed with assistance from AI tools.
"""G1 29-DOF warehouse environment with lidar (operator preset).

USD-based scene import for training in structured environments (warehouse,
booth floor plans). Includes a lidar sensor (RayCasterCfg) for obstacle
detection.

GridPatternCfg(resolution=2.0, size=(240.0, 0.0)) produces 121 rays
(240/2 + 1 = 121 points in a single row). Each ray returns xyz hit
coordinates, giving 121 * 3 = 363 lidar dims.

Obs dim: 103 (base) + 363 (lidar: 121 rays x 3 xyz) = 466.
Action dim: 29 (same as flat).

NOTE: GridPatternCfg interprets size/resolution as spatial (meters), not
angular (degrees). The resulting scan pattern is a linear sweep, not a
uniform angular FOV. Switch to LidarPatternCfg when available in Isaac Lab
for proper angular lidar simulation.
"""

from __future__ import annotations

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from wbc_pipeline.envs.g1_29dof_shared_cfgs import (
    G1_29DOF_ActionsCfg,
    G1_29DOF_CommandsCfg,
    G1_29DOF_CurriculumCfg,
    G1_29DOF_EventsCfg,
    G1_29DOF_RewardsCfg,
    G1_29DOF_TerminationsCfg,
    apply_common_sim_settings,
    make_robot_cfg,
)
from wbc_pipeline.envs.joint_presets import OPERATOR_PRESET
from wbc_pipeline.envs.mdp import lidar_scan, phase_oscillator

# -- Active preset ------------------------------------------------------------
_PRESET = OPERATOR_PRESET

_ROBOT_29DOF = SceneEntityCfg("robot", joint_names=list(_PRESET.joint_order))

# Configurable USD scene path. Defaults to Isaac Sim's sample warehouse.
# Override with WBC_SCENE_USD_PATH env var for custom scenes (booth floor plans, etc.)
WAREHOUSE_USD_PATH = os.environ.get(
    "WBC_SCENE_USD_PATH",
    f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
)

# GridPatternCfg(resolution=2.0, size=(240.0, 0.0)) produces:
#   horizontal: 240/2 + 1 = 121 points, vertical: 0/2 + 1 = 1 row → 121 rays
# lidar_scan returns xyz per ray → 121 * 3 = 363 dims
_LIDAR_NUM_RAYS = 121


# ==============================================================================
# Scene — USD warehouse + lidar
# ==============================================================================


@configclass
class G1_29DOF_WarehouseSceneCfg(InteractiveSceneCfg):
    """Scene using USD warehouse asset with lidar sensor."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=WAREHOUSE_USD_PATH,
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
    # NOTE: GridPatternCfg produces a spatial scan (meters), not angular (degrees).
    # Use LidarPatternCfg when available in Isaac Lab for proper angular lidar.
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=2.0,
            size=(240.0, 0.0),
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


# ==============================================================================
# Observations — 103 base + 121*3 lidar = 466
# ==============================================================================


@configclass
class G1_29DOF_WarehouseObsCfg:
    """Observation groups — 466-dim (103 base + 363 lidar)."""

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
        lidar_obs = ObsTerm(
            func=lidar_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            clip=(-10.0, 10.0),
        )

    policy: PolicyCfg = PolicyCfg()


# ==============================================================================
# Top-level env config
# ==============================================================================


@configclass
class G1_29DOF_WarehouseEnvCfg(ManagerBasedRLEnvCfg):
    """G1 29-DOF warehouse env with lidar observations."""

    EXPECTED_OBS_DIM = 466
    EXPECTED_ACTION_DIM = 29

    scene: G1_29DOF_WarehouseSceneCfg = G1_29DOF_WarehouseSceneCfg(num_envs=1024, env_spacing=5.0)
    observations: G1_29DOF_WarehouseObsCfg = G1_29DOF_WarehouseObsCfg()
    actions: G1_29DOF_ActionsCfg = G1_29DOF_ActionsCfg()
    commands: G1_29DOF_CommandsCfg = G1_29DOF_CommandsCfg()
    rewards: G1_29DOF_RewardsCfg = G1_29DOF_RewardsCfg()
    terminations: G1_29DOF_TerminationsCfg = G1_29DOF_TerminationsCfg()
    events: G1_29DOF_EventsCfg = G1_29DOF_EventsCfg()
    curriculum: G1_29DOF_CurriculumCfg = G1_29DOF_CurriculumCfg()

    def __post_init__(self):
        apply_common_sim_settings(self)
