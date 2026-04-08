# This project was developed with assistance from AI tools.
"""Joint presets for G1 29-DOF locomotion environments.

Pure Python — no Isaac Lab dependencies. Presets define joint ordering,
default positions, action scale, and phase oscillator parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# -- 29 body joints in Unitree G1 URDF order (no fingers) --------------------
# This order matches both the MuJoCo g1_29dof.xml and the robot operator's
# convention: left leg → right leg → waist → left arm → right arm.
_G1_29DOF_JOINT_ORDER: list[str] = [
    # Left leg (0-5)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6-11)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (12-14)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (15-21)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (22-28)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@dataclass(frozen=True)
class JointPreset:
    """Parameterized joint configuration for a G1 29-DOF environment."""

    name: str
    joint_order: tuple[str, ...]
    default_positions: dict[str, float]
    action_scale: float
    phase_freq_hz: float  # 0.0 disables phase oscillator
    phase_offset: float


# -- Operator preset (robotics-rl inference convention) ----------------------
# Validated in Phases 1-3 against OnnxPolicyAction inference code.
OPERATOR_PRESET = JointPreset(
    name="operator",
    joint_order=tuple(_G1_29DOF_JOINT_ORDER),
    default_positions={
        "left_hip_pitch_joint": -0.312,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.669,
        "left_ankle_pitch_joint": -0.363,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.312,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.669,
        "right_ankle_pitch_joint": -0.363,
        "right_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.073,
        "left_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.6,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.6,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    },
    action_scale=0.25,
    phase_freq_hz=1.25,
    phase_offset=math.pi,
)


# -- Isaac Lab stock preset (G1_29DOF_CFG defaults) -------------------------
# From isaaclab_assets/robots/unitree.py G1_29DOF_CFG.init_state.joint_pos.
# Same joint order as operator (URDF natural order); differs in default
# positions, action scale, and phase oscillator settings.
ISAAC_LAB_PRESET = JointPreset(
    name="isaac-lab-default",
    joint_order=tuple(_G1_29DOF_JOINT_ORDER),
    default_positions={
        "left_hip_pitch_joint": -0.10,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.30,
        "left_ankle_pitch_joint": -0.20,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.10,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.30,
        "right_ankle_pitch_joint": -0.20,
        "right_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    },
    action_scale=0.5,
    phase_freq_hz=0.0,
    phase_offset=0.0,
)


PRESETS: dict[str, JointPreset] = {
    "operator": OPERATOR_PRESET,
    "isaac-lab-default": ISAAC_LAB_PRESET,
}
