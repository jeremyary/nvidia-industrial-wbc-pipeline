# This project was developed with assistance from AI tools.
"""Custom observation terms for G1 29-DOF locomotion."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def phase_oscillator(env: ManagerBasedEnv, freq_hz: float = 1.25, offset: float = math.pi) -> torch.Tensor:
    """Gait phase oscillator producing 4-dim observation: [cos(L), cos(R), sin(L), sin(R)]."""
    if freq_hz == 0.0:
        return torch.zeros(env.num_envs, 4, device=env.device)

    if not hasattr(env, "_phase_left"):
        env._phase_left = torch.zeros(env.num_envs, device=env.device)
        env._phase_right = torch.full((env.num_envs,), offset, device=env.device)

    # Advance phase first
    dt = env.step_dt
    freq = freq_hz * 2.0 * math.pi
    env._phase_left = (env._phase_left + freq * dt) % (2.0 * math.pi)
    env._phase_right = (env._phase_right + freq * dt) % (2.0 * math.pi)

    # Reset envs that just started a new episode (overwrites the stale increment)
    reset_mask = env.episode_length_buf == 0
    env._phase_left[reset_mask] = 0.0
    env._phase_right[reset_mask] = offset

    return torch.stack(
        [
            torch.cos(env._phase_left),
            torch.cos(env._phase_right),
            torch.sin(env._phase_left),
            torch.sin(env._phase_right),
        ],
        dim=-1,
    )


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height scan from RayCasterCfg sensor, returns distances below the base."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]


def lidar_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Lidar scan from RayCasterCfg sensor, flattened xyz hit positions per ray."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.ray_hits_w[..., :3].reshape(env.num_envs, -1)
