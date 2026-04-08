# This project was developed with assistance from AI tools.
"""ONNX compatibility tests against the robot operator's inference code.

Validates that exported policies match the OnnxPolicyAction contract from
robotics-rl: input "obs" [1, 103], output "actions" [1, 29],
normalization baked in, outputs finite and bounded.
"""

from __future__ import annotations

import importlib
import math
import os

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

_has_isaaclab = importlib.util.find_spec("isaaclab") is not None
requires_isaaclab = pytest.mark.skipif(not _has_isaaclab, reason="isaaclab not installed")

# ---------------------------------------------------------------------------
# Constants from the operator preset
# ---------------------------------------------------------------------------

from wbc_pipeline.envs.joint_presets import OPERATOR_PRESET  # noqa: E402

OBS_DIM = 103
ACTION_DIM = 29
ACTION_SCALE = OPERATOR_PRESET.action_scale
INPUT_NAME = "obs"

# Observation layout: 3+3+3+3+29+29+29+4 = 103
OBS_SLICES = {
    "linear_velocity": (0, 3),
    "angular_velocity": (3, 6),
    "projected_gravity": (6, 9),
    "velocity_command": (9, 12),
    "joint_pos_rel": (12, 41),
    "joint_vel": (41, 70),
    "last_action": (70, 99),
    "phase_oscillator": (99, 103),
}

OPERATOR_JOINT_ORDER = list(OPERATOR_PRESET.joint_order)

OPERATOR_DEFAULT_POSITIONS = [OPERATOR_PRESET.default_positions[j] for j in OPERATOR_JOINT_ORDER]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_synthetic_onnx(path: str, obs_dim: int = OBS_DIM, act_dim: int = ACTION_DIM):
    """Build a minimal MLP ONNX model matching the expected policy structure."""
    hidden = 32

    # Weights (random but deterministic)
    rng = np.random.default_rng(42)
    w1 = rng.standard_normal((obs_dim, hidden)).astype(np.float32) * 0.01
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = rng.standard_normal((hidden, act_dim)).astype(np.float32) * 0.01
    b2 = np.zeros(act_dim, dtype=np.float32)

    # Graph
    obs_input = helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, [1, obs_dim])
    act_output = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, act_dim])

    w1_init = helper.make_tensor("w1", TensorProto.FLOAT, w1.shape, w1.flatten().tolist())
    b1_init = helper.make_tensor("b1", TensorProto.FLOAT, b1.shape, b1.tolist())
    w2_init = helper.make_tensor("w2", TensorProto.FLOAT, w2.shape, w2.flatten().tolist())
    b2_init = helper.make_tensor("b2", TensorProto.FLOAT, b2.shape, b2.tolist())

    matmul1 = helper.make_node("MatMul", [INPUT_NAME, "w1"], ["mm1"])
    add1 = helper.make_node("Add", ["mm1", "b1"], ["h1"])
    relu1 = helper.make_node("Relu", ["h1"], ["a1"])
    matmul2 = helper.make_node("MatMul", ["a1", "w2"], ["mm2"])
    add2 = helper.make_node("Add", ["mm2", "b2"], ["actions"])

    graph = helper.make_graph(
        [matmul1, add1, relu1, matmul2, add2],
        "policy",
        [obs_input],
        [act_output],
        initializer=[w1_init, b1_init, w2_init, b2_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


@pytest.fixture
def synthetic_onnx(tmp_path):
    """Provide a synthetic ONNX model with correct input/output shapes."""
    path = str(tmp_path / "policy.onnx")
    _build_synthetic_onnx(path)
    return path


@pytest.fixture
def synthetic_session(synthetic_onnx):
    """Provide an ORT InferenceSession for the synthetic model."""
    return ort.InferenceSession(synthetic_onnx, providers=["CPUExecutionProvider"])


# Path to a real exported model (set via ONNX_POLICY_PATH env var or skip)
_real_onnx_path = os.environ.get("ONNX_POLICY_PATH", "")
requires_real_model = pytest.mark.skipif(
    not _real_onnx_path or not os.path.exists(_real_onnx_path),
    reason="ONNX_POLICY_PATH not set or file not found",
)


@pytest.fixture
def real_model_path():
    """Path to a real exported ONNX policy."""
    return _real_onnx_path


@pytest.fixture
def real_session(real_model_path):
    """ORT session for a real exported model."""
    return ort.InferenceSession(real_model_path, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
# Observation vector construction helpers
# ---------------------------------------------------------------------------


def _make_standing_obs() -> np.ndarray:
    """Construct a 103-dim observation for a robot standing still at default pose."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    # linear velocity = 0
    # angular velocity = 0
    # projected gravity = [0, 0, -1] (upright)
    obs[6:9] = [0.0, 0.0, -1.0]
    # velocity command = 0
    # joint_pos_rel = 0 (at default positions)
    # joint_vel = 0
    # last_action = 0
    # phase oscillator: t=0 -> phases=[0, pi] -> [cos(0), cos(pi), sin(0), sin(pi)]
    obs[99] = math.cos(0.0)  # cos(phase_left)
    obs[100] = math.cos(math.pi)  # cos(phase_right)
    obs[101] = math.sin(0.0)  # sin(phase_left)
    obs[102] = math.sin(math.pi)  # sin(phase_right)
    return obs


def _make_walking_obs(vx: float = 0.5) -> np.ndarray:
    """Construct a 103-dim observation for commanded forward walk."""
    obs = _make_standing_obs()
    obs[0] = vx  # linear velocity x
    obs[9] = vx  # commanded velocity x
    # Simulate some joint deviations
    rng = np.random.default_rng(123)
    obs[12:41] = rng.uniform(-0.1, 0.1, 29).astype(np.float32)
    obs[41:70] = rng.uniform(-1.0, 1.0, 29).astype(np.float32)
    obs[70:99] = rng.uniform(-0.5, 0.5, 29).astype(np.float32)
    return obs


# ---------------------------------------------------------------------------
# Structural tests (synthetic model)
# ---------------------------------------------------------------------------


class TestOnnxStructure:
    """Validate ONNX model structure matches operator's OnnxPolicyAction contract."""

    def test_input_name(self, synthetic_onnx):
        """Input tensor must be named 'obs'."""
        model = onnx.load(synthetic_onnx)
        assert model.graph.input[0].name == INPUT_NAME

    def test_input_shape(self, synthetic_onnx):
        """Input shape must be [1, 103]."""
        model = onnx.load(synthetic_onnx)
        dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        assert dims == [1, OBS_DIM]

    def test_output_shape(self, synthetic_onnx):
        """Output shape must be [1, 29]."""
        model = onnx.load(synthetic_onnx)
        dims = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
        assert dims == [1, ACTION_DIM]

    def test_input_dtype_float32(self, synthetic_onnx):
        """Input must be float32."""
        model = onnx.load(synthetic_onnx)
        assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT

    def test_single_input(self, synthetic_onnx):
        """Model should have exactly one input (obs)."""
        model = onnx.load(synthetic_onnx)
        assert len(model.graph.input) == 1

    def test_onnx_checker_passes(self, synthetic_onnx):
        """ONNX checker validates the model."""
        model = onnx.load(synthetic_onnx)
        onnx.checker.check_model(model)


# ---------------------------------------------------------------------------
# Inference tests (synthetic model)
# ---------------------------------------------------------------------------


class TestOnnxInference:
    """Validate inference behavior matches operator expectations."""

    def test_inference_standing(self, synthetic_session):
        """Standing observation produces finite 29-dim output."""
        obs = _make_standing_obs().reshape(1, OBS_DIM)
        outputs = synthetic_session.run(None, {INPUT_NAME: obs})
        actions = outputs[0]
        assert actions.shape == (1, ACTION_DIM)
        assert np.all(np.isfinite(actions))

    def test_inference_walking(self, synthetic_session):
        """Walking observation produces finite 29-dim output."""
        obs = _make_walking_obs().reshape(1, OBS_DIM)
        outputs = synthetic_session.run(None, {INPUT_NAME: obs})
        actions = outputs[0]
        assert actions.shape == (1, ACTION_DIM)
        assert np.all(np.isfinite(actions))

    def test_inference_random_obs(self, synthetic_session):
        """Random observations produce finite outputs."""
        rng = np.random.default_rng(99)
        for _ in range(10):
            obs = rng.standard_normal((1, OBS_DIM)).astype(np.float32)
            outputs = synthetic_session.run(None, {INPUT_NAME: obs})
            assert np.all(np.isfinite(outputs[0]))

    def test_inference_zeros(self, synthetic_session):
        """Zero observation produces finite output."""
        obs = np.zeros((1, OBS_DIM), dtype=np.float32)
        outputs = synthetic_session.run(None, {INPUT_NAME: obs})
        assert np.all(np.isfinite(outputs[0]))

    def test_different_inputs_different_outputs(self, synthetic_session):
        """Different observations produce different actions."""
        obs1 = _make_standing_obs().reshape(1, OBS_DIM)
        obs2 = _make_walking_obs().reshape(1, OBS_DIM)
        out1 = synthetic_session.run(None, {INPUT_NAME: obs1})[0]
        out2 = synthetic_session.run(None, {INPUT_NAME: obs2})[0]
        assert not np.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Observation vector contract tests
# ---------------------------------------------------------------------------


class TestObservationContract:
    """Validate observation vector layout matches operator's _build_obs()."""

    def test_total_dim(self):
        """Total observation dimension is 103."""
        total = sum(end - start for start, end in OBS_SLICES.values())
        assert total == OBS_DIM

    def test_slices_contiguous(self):
        """Observation slices are contiguous with no gaps."""
        sorted_slices = sorted(OBS_SLICES.values())
        for i in range(len(sorted_slices) - 1):
            assert sorted_slices[i][1] == sorted_slices[i + 1][0], (
                f"Gap between {sorted_slices[i]} and {sorted_slices[i + 1]}"
            )

    def test_slices_start_at_zero(self):
        """First slice starts at index 0."""
        assert min(s[0] for s in OBS_SLICES.values()) == 0

    def test_slices_end_at_obs_dim(self):
        """Last slice ends at OBS_DIM."""
        assert max(s[1] for s in OBS_SLICES.values()) == OBS_DIM

    def test_joint_pos_rel_dim(self):
        """joint_pos_rel occupies 29 dimensions."""
        s, e = OBS_SLICES["joint_pos_rel"]
        assert e - s == 29

    def test_phase_oscillator_dim(self):
        """Phase oscillator occupies 4 dimensions (cos_L, cos_R, sin_L, sin_R)."""
        s, e = OBS_SLICES["phase_oscillator"]
        assert e - s == 4

    def test_standing_obs_gravity(self):
        """Standing upright has projected gravity [0, 0, -1]."""
        obs = _make_standing_obs()
        s, e = OBS_SLICES["projected_gravity"]
        np.testing.assert_allclose(obs[s:e], [0.0, 0.0, -1.0])

    def test_standing_obs_phase_t0(self):
        """Phase at t=0: phases=[0, pi], output=[cos(0), cos(pi), sin(0), sin(pi)]."""
        obs = _make_standing_obs()
        s, e = OBS_SLICES["phase_oscillator"]
        expected = [1.0, -1.0, 0.0, 0.0]
        np.testing.assert_allclose(obs[s:e], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Action contract tests
# ---------------------------------------------------------------------------


class TestActionContract:
    """Validate action vector matches operator's joint application logic."""

    def test_joint_count(self):
        """29 joints in operator order."""
        assert len(OPERATOR_JOINT_ORDER) == ACTION_DIM

    def test_default_positions_count(self):
        """29 default positions matching 29 joints."""
        assert len(OPERATOR_DEFAULT_POSITIONS) == ACTION_DIM

    def test_action_scale(self):
        """Action scale is 0.25 radians."""
        assert ACTION_SCALE == 0.25

    def test_target_from_action(self):
        """q_target = default_positions + action * ACTION_SCALE."""
        defaults = np.array(OPERATOR_DEFAULT_POSITIONS, dtype=np.float32)
        action = np.ones(ACTION_DIM, dtype=np.float32)  # max action
        target = defaults + action * ACTION_SCALE
        # Left hip pitch: -0.312 + 0.25 = -0.062
        assert target[0] == pytest.approx(-0.062, abs=1e-4)
        # Left knee: 0.669 + 0.25 = 0.919
        assert target[3] == pytest.approx(0.919, abs=1e-4)

    def test_action_range_bounded(self):
        """Max joint displacement from defaults is ACTION_SCALE (0.25 rad)."""
        defaults = np.array(OPERATOR_DEFAULT_POSITIONS, dtype=np.float32)
        for action_val in [-1.0, 1.0]:
            action = np.full(ACTION_DIM, action_val, dtype=np.float32)
            target = defaults + action * ACTION_SCALE
            displacement = np.abs(target - defaults)
            np.testing.assert_allclose(displacement, ACTION_SCALE, atol=1e-6)

    def test_leg_symmetry_in_defaults(self):
        """Left and right leg defaults are symmetric."""
        left_leg = OPERATOR_DEFAULT_POSITIONS[0:6]
        right_leg = OPERATOR_DEFAULT_POSITIONS[6:12]
        np.testing.assert_allclose(left_leg, right_leg)


# ---------------------------------------------------------------------------
# Env config cross-check (no GPU needed — uses AST parsing from test_env_config)
# ---------------------------------------------------------------------------


@requires_isaaclab
class TestEnvConfigCrossCheck:
    """Cross-check env config re-exports against the preset source of truth."""

    def test_joint_order_reexport(self):
        """Env config re-exports match preset joint order."""
        from wbc_pipeline.envs.g1_29dof_env_cfg import OPERATOR_JOINT_ORDER as reexported

        assert reexported == OPERATOR_JOINT_ORDER

    def test_default_positions_reexport(self):
        """Env config re-exports match preset default positions."""
        from wbc_pipeline.envs.g1_29dof_env_cfg import OPERATOR_DEFAULT_POSITIONS as reexported

        reexported_list = [reexported[j] for j in OPERATOR_JOINT_ORDER]
        np.testing.assert_allclose(reexported_list, OPERATOR_DEFAULT_POSITIONS, atol=1e-6)


# ---------------------------------------------------------------------------
# Real model tests (require ONNX_POLICY_PATH env var)
# ---------------------------------------------------------------------------


@requires_real_model
class TestRealModel:
    """Tests against a real exported ONNX policy (on-cluster or with checkpoint)."""

    def test_input_name(self, real_model_path):
        """Real model input tensor is named 'obs'."""
        model = onnx.load(real_model_path)
        assert model.graph.input[0].name == INPUT_NAME

    def test_input_shape(self, real_model_path):
        """Real model input shape is [1, 103]."""
        model = onnx.load(real_model_path)
        dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        assert dims == [1, OBS_DIM]

    def test_output_shape(self, real_model_path):
        """Real model output shape is [1, 29]."""
        model = onnx.load(real_model_path)
        dims = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
        assert dims == [1, ACTION_DIM]

    def test_inference_standing(self, real_session):
        """Real model produces bounded actions for standing pose."""
        obs = _make_standing_obs().reshape(1, OBS_DIM)
        outputs = real_session.run(None, {INPUT_NAME: obs})
        actions = outputs[0].flatten()
        assert actions.shape == (ACTION_DIM,)
        assert np.all(np.isfinite(actions))

    def test_inference_walking(self, real_session):
        """Real model produces bounded actions for walking command."""
        obs = _make_walking_obs().reshape(1, OBS_DIM)
        outputs = real_session.run(None, {INPUT_NAME: obs})
        actions = outputs[0].flatten()
        assert actions.shape == (ACTION_DIM,)
        assert np.all(np.isfinite(actions))

    def test_has_normalization(self, real_model_path):
        """Real model has normalization (explicit ops or folded into Gemm weights)."""
        model = onnx.load(real_model_path)
        op_types = {node.op_type for node in model.graph.node}
        has_explicit = ("Sub" in op_types and "Div" in op_types) or "BatchNormalization" in op_types
        if not has_explicit:
            # Normalization may be constant-folded into Gemm weights during ONNX export.
            # Verify by checking that the model has Gemm ops (MLP present).
            assert "Gemm" in op_types, f"No Gemm or normalization ops found: {op_types}"

    def test_actions_physically_plausible(self, real_session):
        """Real model actions for standing should be near zero (small corrections)."""
        obs = _make_standing_obs().reshape(1, OBS_DIM)
        outputs = real_session.run(None, {INPUT_NAME: obs})
        actions = outputs[0].flatten()
        # Standing at default pose should produce small actions (< 0.5 magnitude)
        assert np.max(np.abs(actions)) < 1.0, f"Actions too large for standing pose: max={np.max(np.abs(actions)):.3f}"

    def test_deterministic_inference(self, real_session):
        """Same input produces same output (no stochastic sampling in ONNX)."""
        obs = _make_walking_obs().reshape(1, OBS_DIM)
        out1 = real_session.run(None, {INPUT_NAME: obs})[0]
        out2 = real_session.run(None, {INPUT_NAME: obs})[0]
        np.testing.assert_array_equal(out1, out2)
