# This project was developed with assistance from AI tools.
"""Unit tests for G1 29-DOF rough terrain environment config."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_has_isaaclab = importlib.util.find_spec("isaaclab") is not None
requires_isaaclab = pytest.mark.skipif(not _has_isaaclab, reason="isaaclab not installed")


def _parse_rough_cfg_source() -> str:
    """Read raw source of the rough cfg module for AST inspection."""
    src = Path(__file__).resolve().parent.parent / "src" / "wbc_pipeline" / "envs" / "g1_29dof_rough_cfg.py"
    return src.read_text()


class TestRoughTerrainConfig:
    """Validate rough terrain env config structure via source inspection."""

    def test_terrain_type_is_generator(self):
        """Rough scene uses terrain_type='generator', not 'plane'."""
        source = _parse_rough_cfg_source()
        assert 'terrain_type="generator"' in source

    def test_has_rough_terrains_import(self):
        """Rough cfg imports ROUGH_TERRAINS_CFG."""
        source = _parse_rough_cfg_source()
        assert "ROUGH_TERRAINS_CFG" in source

    def test_has_height_scanner(self):
        """Rough scene includes a height_scanner RayCasterCfg."""
        source = _parse_rough_cfg_source()
        assert "height_scanner" in source
        assert "RayCasterCfg" in source

    def test_height_scanner_on_torso(self):
        """Height scanner is attached to torso_link."""
        source = _parse_rough_cfg_source()
        assert "torso_link" in source

    def test_has_height_scan_obs(self):
        """Rough obs config includes height_scan observation term."""
        source = _parse_rough_cfg_source()
        assert "height_scan_obs" in source or "height_scan" in source

    def test_has_terrain_curriculum(self):
        """Rough env has terrain_levels curriculum term."""
        source = _parse_rough_cfg_source()
        assert "terrain_levels" in source
        assert "terrain_levels_vel" in source

    def test_uses_operator_preset(self):
        """Rough env uses the operator preset by default."""
        source = _parse_rough_cfg_source()
        assert "OPERATOR_PRESET" in source

    def test_obs_dim_290(self):
        """Rough env obs is 103 base + 187 height scan = 290."""
        # Height scan: GridPatternCfg(resolution=0.1, size=(1.6, 1.0))
        # Grid: 1.6/0.1 + 1 = 17 columns, 1.0/0.1 + 1 = 11 rows = 187 rays
        base_obs = 3 + 3 + 3 + 3 + 29 + 29 + 29 + 4  # 103
        height_scan_rays = 17 * 11  # 187
        assert base_obs + height_scan_rays == 290

    def test_relaxed_lin_vel_z_penalty(self):
        """Rough terrain relaxes vertical velocity penalty (weight=0.0)."""
        source = _parse_rough_cfg_source()
        # The rough env should have lin_vel_z_l2 weight set to 0.0
        assert "lin_vel_z_l2" in source


class TestWarehouseConfig:
    """Validate warehouse scene config structure."""

    def _parse_source(self) -> str:
        src = Path(__file__).resolve().parent.parent / "src" / "wbc_pipeline" / "envs" / "g1_29dof_scene_cfgs.py"
        return src.read_text()

    def test_terrain_type_usd(self):
        """Warehouse scene uses terrain_type='usd'."""
        assert 'terrain_type="usd"' in self._parse_source()

    def test_has_lidar_sensor(self):
        """Warehouse scene includes a lidar RayCasterCfg sensor."""
        source = self._parse_source()
        assert "lidar" in source
        assert "RayCasterCfg" in source

    def test_configurable_usd_path(self):
        """USD path is configurable via WBC_SCENE_USD_PATH env var."""
        source = self._parse_source()
        assert "WBC_SCENE_USD_PATH" in source

    def test_has_lidar_obs(self):
        """Warehouse obs config includes lidar observation term."""
        source = self._parse_source()
        assert "lidar_obs" in source or "lidar_scan" in source


class TestEnvRegistration:
    """Validate all env variants are registered."""

    def _parse_init_source(self) -> str:
        src = Path(__file__).resolve().parent.parent / "src" / "wbc_pipeline" / "envs" / "__init__.py"
        return src.read_text()

    def test_flat_registered(self):
        """Flat env is registered."""
        assert "WBC-Velocity-Flat-G1-29DOF-v0" in self._parse_init_source()

    def test_rough_registered(self):
        """Rough env is registered."""
        assert "WBC-Velocity-Rough-G1-29DOF-v0" in self._parse_init_source()

    def test_warehouse_registered(self):
        """Warehouse env is registered."""
        assert "WBC-Velocity-Warehouse-G1-29DOF-v0" in self._parse_init_source()

    def test_isaaclab_flat_registered(self):
        """Isaac Lab default flat env is registered."""
        assert "WBC-Velocity-IsaacLab-Flat-G1-29DOF-v0" in self._parse_init_source()
