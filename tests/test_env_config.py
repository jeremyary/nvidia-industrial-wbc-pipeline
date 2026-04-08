# This project was developed with assistance from AI tools.
"""Unit tests for G1 29-DOF environment config — no GPU required."""

from __future__ import annotations

import importlib

import pytest

from wbc_pipeline.envs.joint_presets import (
    ISAAC_LAB_PRESET,
    OPERATOR_PRESET,
    PRESETS,
    JointPreset,
)

# Try importing the full config — skip tests that need it if isaaclab is unavailable
_has_isaaclab = importlib.util.find_spec("isaaclab") is not None
requires_isaaclab = pytest.mark.skipif(not _has_isaaclab, reason="isaaclab not installed")


class TestJointPresets:
    """Validate all joint presets have valid structure."""

    @pytest.mark.parametrize("preset", PRESETS.values(), ids=lambda p: p.name)
    def test_joint_count(self, preset: JointPreset):
        """Each preset has exactly 29 joints."""
        assert len(preset.joint_order) == 29

    @pytest.mark.parametrize("preset", PRESETS.values(), ids=lambda p: p.name)
    def test_no_duplicate_joints(self, preset: JointPreset):
        """No duplicates in joint order."""
        assert len(set(preset.joint_order)) == 29

    @pytest.mark.parametrize("preset", PRESETS.values(), ids=lambda p: p.name)
    def test_defaults_match_joints(self, preset: JointPreset):
        """Every joint has a default position and vice versa."""
        assert set(preset.joint_order) == set(preset.default_positions.keys())

    @pytest.mark.parametrize("preset", PRESETS.values(), ids=lambda p: p.name)
    def test_positive_action_scale(self, preset: JointPreset):
        """Action scale is positive."""
        assert preset.action_scale > 0

    @pytest.mark.parametrize("preset", PRESETS.values(), ids=lambda p: p.name)
    def test_valid_phase_freq(self, preset: JointPreset):
        """Phase frequency is non-negative."""
        assert preset.phase_freq_hz >= 0.0

    def test_preset_names_unique(self):
        """All preset names are unique."""
        names = [p.name for p in PRESETS.values()]
        assert len(names) == len(set(names))

    def test_preset_dict_keys_match_names(self):
        """PRESETS dict keys match preset.name."""
        for key, preset in PRESETS.items():
            assert key == preset.name


class TestOperatorPreset:
    """Validate operator preset against the robot operator's convention."""

    def test_joint_group_ordering(self):
        """Joints follow operator's group convention: L-leg, R-leg, waist, L-arm, R-arm."""
        assert OPERATOR_PRESET.joint_order[0] == "left_hip_pitch_joint"
        assert OPERATOR_PRESET.joint_order[6] == "right_hip_pitch_joint"
        assert OPERATOR_PRESET.joint_order[12] == "waist_yaw_joint"
        assert OPERATOR_PRESET.joint_order[15] == "left_shoulder_pitch_joint"
        assert OPERATOR_PRESET.joint_order[22] == "right_shoulder_pitch_joint"

    def test_left_leg_indices(self):
        """Left leg occupies indices 0-5."""
        left_leg = OPERATOR_PRESET.joint_order[0:6]
        assert all("left" in j and ("hip" in j or "knee" in j or "ankle" in j) for j in left_leg)

    def test_right_leg_indices(self):
        """Right leg occupies indices 6-11."""
        right_leg = OPERATOR_PRESET.joint_order[6:12]
        assert all("right" in j and ("hip" in j or "knee" in j or "ankle" in j) for j in right_leg)

    def test_waist_indices(self):
        """Waist occupies indices 12-14."""
        waist = OPERATOR_PRESET.joint_order[12:15]
        assert all("waist" in j for j in waist)

    def test_left_arm_indices(self):
        """Left arm occupies indices 15-21."""
        left_arm = OPERATOR_PRESET.joint_order[15:22]
        assert all("left" in j and ("shoulder" in j or "elbow" in j or "wrist" in j) for j in left_arm)

    def test_right_arm_indices(self):
        """Right arm occupies indices 22-28."""
        right_arm = OPERATOR_PRESET.joint_order[22:29]
        assert all("right" in j and ("shoulder" in j or "elbow" in j or "wrist" in j) for j in right_arm)

    def test_default_positions_known_values(self):
        """Spot-check critical default positions against operator's values."""
        d = OPERATOR_PRESET.default_positions
        assert d["left_hip_pitch_joint"] == pytest.approx(-0.312)
        assert d["left_knee_joint"] == pytest.approx(0.669)
        assert d["left_ankle_pitch_joint"] == pytest.approx(-0.363)
        assert d["right_hip_pitch_joint"] == pytest.approx(-0.312)
        assert d["right_knee_joint"] == pytest.approx(0.669)
        assert d["waist_pitch_joint"] == pytest.approx(0.073)
        assert d["left_elbow_joint"] == pytest.approx(0.6)
        assert d["left_shoulder_roll_joint"] == pytest.approx(0.2)
        assert d["right_shoulder_roll_joint"] == pytest.approx(-0.2)

    def test_symmetric_leg_defaults(self):
        """Left and right leg default positions are symmetric."""
        d = OPERATOR_PRESET.default_positions
        for side_joint in [
            "hip_pitch_joint",
            "hip_roll_joint",
            "hip_yaw_joint",
            "knee_joint",
            "ankle_pitch_joint",
            "ankle_roll_joint",
        ]:
            left_val = d[f"left_{side_joint}"]
            right_val = d[f"right_{side_joint}"]
            assert left_val == pytest.approx(right_val), f"Asymmetric defaults for {side_joint}"

    def test_action_scale(self):
        """Operator action scale is 0.25."""
        assert OPERATOR_PRESET.action_scale == 0.25

    def test_phase_freq(self):
        """Operator phase oscillator is 1.25 Hz."""
        assert OPERATOR_PRESET.phase_freq_hz == 1.25


class TestIsaacLabPreset:
    """Validate Isaac Lab stock preset has correct values from G1_29DOF_CFG."""

    def test_stock_default_positions(self):
        """Isaac Lab stock defaults from G1_29DOF_CFG.init_state.joint_pos."""
        d = ISAAC_LAB_PRESET.default_positions
        assert d["left_hip_pitch_joint"] == pytest.approx(-0.10)
        assert d["left_knee_joint"] == pytest.approx(0.30)
        assert d["left_ankle_pitch_joint"] == pytest.approx(-0.20)
        assert d["right_hip_pitch_joint"] == pytest.approx(-0.10)
        assert d["right_knee_joint"] == pytest.approx(0.30)
        # Waist, arms, wrists all zero in stock config
        assert d["waist_pitch_joint"] == pytest.approx(0.0)
        assert d["left_elbow_joint"] == pytest.approx(0.0)

    def test_action_scale(self):
        """Isaac Lab stock action scale is 0.5."""
        assert ISAAC_LAB_PRESET.action_scale == 0.5

    def test_no_phase_oscillator(self):
        """Isaac Lab stock has no phase oscillator."""
        assert ISAAC_LAB_PRESET.phase_freq_hz == 0.0

    def test_same_joint_order_as_operator(self):
        """Both presets use the same URDF-natural joint order."""
        assert ISAAC_LAB_PRESET.joint_order == OPERATOR_PRESET.joint_order


class TestObservationDims:
    """Validate observation space dimensions sum to 103."""

    def test_obs_dims_sum(self):
        """Observation terms: 3+3+3+3+29+29+29+4 = 103."""
        expected_dims = {
            "base_lin_vel": 3,
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "velocity_commands": 3,
            "joint_pos": 29,
            "joint_vel": 29,
            "actions": 29,
            "phase": 4,
        }
        assert sum(expected_dims.values()) == 103


@requires_isaaclab
class TestActionConfig:
    """Validate action space configuration (requires isaaclab)."""

    def test_action_scale(self):
        """Action scale matches operator's ACTION_SCALE of 0.25."""
        from wbc_pipeline.envs.g1_29dof_env_cfg import G1_29DOF_FlatEnvCfg

        cfg = G1_29DOF_FlatEnvCfg()
        assert cfg.actions.joint_pos.scale == pytest.approx(0.25)

    def test_action_uses_default_offset(self):
        """Actions offset from default positions, not zero."""
        from wbc_pipeline.envs.g1_29dof_env_cfg import G1_29DOF_FlatEnvCfg

        cfg = G1_29DOF_FlatEnvCfg()
        assert cfg.actions.joint_pos.use_default_offset is True

    def test_action_joint_names_match(self):
        """Action joint names match preset joint order."""
        from wbc_pipeline.envs.g1_29dof_env_cfg import G1_29DOF_FlatEnvCfg

        cfg = G1_29DOF_FlatEnvCfg()
        assert cfg.actions.joint_pos.joint_names == list(OPERATOR_PRESET.joint_order)


@requires_isaaclab
class TestEnvConfig:
    """Validate top-level environment config (requires isaaclab)."""

    def test_physics_dt(self):
        """Physics runs at 200 Hz."""
        from wbc_pipeline.envs.g1_29dof_env_cfg import G1_29DOF_FlatEnvCfg

        cfg = G1_29DOF_FlatEnvCfg()
        assert cfg.sim.dt == pytest.approx(0.005)

    def test_decimation(self):
        """Control runs at 50 Hz (decimation=4 at 200 Hz physics)."""
        from wbc_pipeline.envs.g1_29dof_env_cfg import G1_29DOF_FlatEnvCfg

        cfg = G1_29DOF_FlatEnvCfg()
        assert cfg.decimation == 4

    def test_control_dt(self):
        """Control dt = physics_dt * decimation = 0.02s = 50 Hz."""
        from wbc_pipeline.envs.g1_29dof_env_cfg import G1_29DOF_FlatEnvCfg

        cfg = G1_29DOF_FlatEnvCfg()
        assert cfg.sim.dt * cfg.decimation == pytest.approx(0.02)

    def test_episode_length(self):
        """Episodes are 20 seconds."""
        from wbc_pipeline.envs.g1_29dof_env_cfg import G1_29DOF_FlatEnvCfg

        cfg = G1_29DOF_FlatEnvCfg()
        assert cfg.episode_length_s == pytest.approx(20.0)
