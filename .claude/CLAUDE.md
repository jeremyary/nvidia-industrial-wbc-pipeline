# Project Conventions

## Architecture
- Isaac Lab RL training pipeline for Unitree G1 29-DOF humanoid robot
- Container: Isaac Sim 5.1.0 + Isaac Lab v2.3.2 + RSL-RL
- Deployment: RHOAI on OpenShift with B200 GPUs (HyperShift) or L40S (existing cluster)
- Pipeline orchestration: KFP v2 on RHOAI DSPA
- Output: ONNX policies compatible with robotics-rl inference code

## Critical Compatibility Target
- ONNX policies must have 103-dim input, 29-dim output
- Joint ordering must match actions/joints.py in robotics-rl
- Default positions must match DEFAULT_POSITIONS exactly
- Observation normalization baked into ONNX graph

## Container
- Registry: quay.io/jary/
- Runtime: Podman with CDI (`--device nvidia.com/gpu=all`, NOT `--gpus all`)
- Base: nvcr.io/nvidia/isaac-sim:5.1.0

## Testing
- Use `respx` for mocking `httpx` in async tests.
- Bind mock/test servers to `127.0.0.1`, not `0.0.0.0`.
