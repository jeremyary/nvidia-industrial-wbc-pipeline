# This project was developed with assistance from AI tools.
"""Unit tests for the phase oscillator observation term."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from wbc_pipeline.envs.mdp.observations import phase_oscillator


def _make_mock_env(num_envs: int = 4, dt: float = 0.02) -> SimpleNamespace:
    """Create a minimal mock env with the attributes phase_oscillator needs."""
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=dt,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    )


class TestPhaseOscillatorShape:
    """Validate output shape and dtype."""

    def test_output_shape(self):
        """Returns [num_envs, 4] tensor."""
        env = _make_mock_env(num_envs=8)
        result = phase_oscillator(env)
        assert result.shape == (8, 4)

    def test_output_dtype(self):
        """Output is float32."""
        env = _make_mock_env()
        result = phase_oscillator(env)
        assert result.dtype == torch.float32


class TestPhaseOscillatorDisabled:
    """Validate behavior when freq_hz=0 (disabled)."""

    def test_returns_zeros(self):
        """Disabled oscillator returns all zeros."""
        env = _make_mock_env()
        result = phase_oscillator(env, freq_hz=0.0)
        assert torch.all(result == 0)

    def test_shape_preserved(self):
        """Shape is [num_envs, 4] even when disabled."""
        env = _make_mock_env(num_envs=16)
        result = phase_oscillator(env, freq_hz=0.0)
        assert result.shape == (16, 4)


class TestPhaseOscillatorReset:
    """Validate reset behavior."""

    def test_first_call_returns_initial_phase(self):
        """First call (all envs just reset) returns [cos(0), cos(pi), sin(0), sin(pi)]."""
        env = _make_mock_env()
        result = phase_oscillator(env)
        expected = torch.tensor([1.0, -1.0, 0.0, 0.0])
        torch.testing.assert_close(result[0], expected, atol=1e-6, rtol=1e-6)

    def test_reset_restores_initial_phase(self):
        """After stepping and resetting, phase returns to initial values."""
        env = _make_mock_env(num_envs=2)

        # Step several times (advance phase)
        for step in range(10):
            env.episode_length_buf[:] = step + 1
            phase_oscillator(env)

        # Reset env 0 only
        env.episode_length_buf[0] = 0
        env.episode_length_buf[1] = 11
        result = phase_oscillator(env)

        # Env 0 should be at initial phase [cos(0), cos(pi), sin(0), sin(pi)]
        expected = torch.tensor([1.0, -1.0, 0.0, 0.0])
        torch.testing.assert_close(result[0], expected, atol=1e-6, rtol=1e-6)

        # Env 1 should NOT be at initial phase (it kept advancing)
        assert not torch.allclose(result[1], expected, atol=1e-3)


class TestPhaseOscillatorAdvancement:
    """Validate phase advancement behavior."""

    def test_phase_advances_each_step(self):
        """Phase changes between consecutive steps."""
        env = _make_mock_env(num_envs=1)
        r1 = phase_oscillator(env).clone()
        env.episode_length_buf[:] = 1
        r2 = phase_oscillator(env).clone()
        assert not torch.allclose(r1, r2)

    def test_left_right_offset(self):
        """Left and right phases are offset by pi."""
        env = _make_mock_env(num_envs=1)
        result = phase_oscillator(env, offset=math.pi)
        # cos(0)=1, cos(pi)=-1, sin(0)=0, sin(pi)~=0
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)  # cos(left=0)
        assert result[0, 1] == pytest.approx(-1.0, abs=1e-6)  # cos(right=pi)

    def test_custom_offset(self):
        """Custom offset shifts right phase relative to left."""
        env = _make_mock_env(num_envs=1)
        result = phase_oscillator(env, offset=math.pi / 2)
        # left phase = 0, right phase = pi/2
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)  # cos(0)
        assert result[0, 1] == pytest.approx(0.0, abs=1e-6)  # cos(pi/2)
        assert result[0, 2] == pytest.approx(0.0, abs=1e-6)  # sin(0)
        assert result[0, 3] == pytest.approx(1.0, abs=1e-6)  # sin(pi/2)

    def test_phase_wraps(self):
        """Phase stays in [0, 2*pi] after many steps."""
        env = _make_mock_env(num_envs=1, dt=0.02)
        for step in range(1000):
            env.episode_length_buf[:] = step
            phase_oscillator(env, freq_hz=1.25)
        # Phase should be bounded
        assert 0.0 <= env._phase_left[0].item() < 2.0 * math.pi
        assert 0.0 <= env._phase_right[0].item() < 2.0 * math.pi

    def test_output_bounded(self):
        """cos/sin outputs are always in [-1, 1]."""
        env = _make_mock_env(num_envs=4, dt=0.02)
        for step in range(100):
            env.episode_length_buf[:] = step
            result = phase_oscillator(env, freq_hz=1.25)
            assert torch.all(result >= -1.0)
            assert torch.all(result <= 1.0)
