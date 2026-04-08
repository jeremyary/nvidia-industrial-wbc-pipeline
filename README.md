# WBC Pipeline — G1 29-DOF Locomotion Training

> [!NOTE]
> This project was developed with assistance from AI tools.

RL training pipeline for the Unitree G1 humanoid robot (29-DOF, no fingers) using Isaac Lab v2.3.2 + RSL-RL PPO. Trains walking policies and exports them as ONNX models with observation normalization baked in, ready for real-time inference on the physical robot.

## Why This Exists

Isaac Lab ships G1 environments for the 37-DOF variant (body + fingers). The downstream inference stack expects 29-DOF policies with a specific joint ordering, default positions, and 103-dim observation vector. This pipeline bridges that gap — custom 29-DOF environments that produce ONNX models matching the inference contract exactly.

## Environments

Four registered gymnasium environments, all sharing the same 29-joint action space:

| Task ID | Obs Dim | Terrain | Sensors | Preset |
|---------|---------|---------|---------|--------|
| `WBC-Velocity-Flat-G1-29DOF-v0` | 103 | Flat plane | Contact | Operator |
| `WBC-Velocity-Rough-G1-29DOF-v0` | 290 | Procedural rough | Contact + height scan (187 rays) | Operator |
| `WBC-Velocity-Warehouse-G1-29DOF-v0` | 466 | USD warehouse scene | Contact + lidar (121 rays x 3) | Operator |
| `WBC-Velocity-IsaacLab-Flat-G1-29DOF-v0` | 103 | Flat plane | Contact | Isaac Lab stock |

**Observations** (103-dim base): linear velocity (3) + angular velocity (3) + projected gravity (3) + velocity commands (3) + joint positions (29) + joint velocities (29) + last actions (29) + phase oscillator (4). Rough and warehouse envs append sensor data.

**Joint presets** (`joint_presets.py`) define the joint ordering, default positions, action scale, and phase oscillator settings. The "operator" preset matches the inference stack. The "Isaac Lab stock" preset uses upstream defaults (shallower knee bend, `action_scale=0.5`, no phase oscillator). Adding a new preset (e.g., for SONIC/BONES) means adding a `JointPreset` dataclass instance — no config classes need to change.

## ONNX Contract

Exported policies must satisfy:

- Input: `"obs"` tensor, shape `[1, obs_dim]`, float32
- Output: `"actions"` tensor, shape `[1, 29]`, float32
- Observation normalization baked into the ONNX graph (not applied at inference time)
- Joint ordering matches `OPERATOR_PRESET.joint_order`
- Action = position offset scaled by `action_scale` (0.25 for operator, 0.5 for Isaac Lab stock)

The `export_onnx.py` script derives dimensions from the live environment and validates the exported model against `EXPECTED_OBS_DIM` / `EXPECTED_ACTION_DIM` declared on each env config.

## Container

Built on `nvcr.io/nvidia/isaac-sim:5.1.0` with Isaac Lab v2.3.2 cloned and installed inside. Two-stage Containerfile: `isaaclab-base` (cached, ~25 min build) and `runtime` (rebuilds in seconds when only `src/` changes).

```bash
# Requires NGC API key in .env (see .env.example)
make build          # podman build with CDI
make push           # push to quay.io
make smoke-test     # local GPU test (10 iters, 64 envs)
```

Runs on OpenShift with NVIDIA GPU operator. Requires `anyuid` SCC (Isaac Sim needs root).

## Training

```bash
# Inside the container or via job manifest:
python -m wbc_pipeline.train \
  --task WBC-Velocity-Flat-G1-29DOF-v0 \
  --headless --num_envs 4096 --max_iterations 6000

# Export trained policy:
python -m wbc_pipeline.export_onnx \
  --task WBC-Velocity-Flat-G1-29DOF-v0 \
  --headless \
  --checkpoint /path/to/model_6000.pt \
  --output_dir /tmp/onnx
```

**Infrastructure integrations** (all opt-in via env vars):
- **S3 checkpointing**: Set `S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`. Checkpoints upload every N iterations and resume with `--resume s3`.
- **MLflow tracking**: Set `MLFLOW_TRACKING_URI`. Logs mean reward, episode length, and learning rate per iteration.
- **SIGTERM handling**: Saves a checkpoint before exit on preemption.

## Deployment

Kubernetes manifests in `deploy/` for OpenShift + RHOAI:

- `namespace.yaml` — creates `wbc-training` namespace
- `gpu-scc.yaml` — ServiceAccount + anyuid ClusterRoleBinding
- `minio.yaml` + `minio-init-job.yaml` — S3-compatible storage for checkpoints
- `mlflow.yaml` + `mlflow-rbac.yaml` — MLflow experiment tracking RBAC
- `smoke-test-job.yaml` through `overnight-training-job.yaml` — GPU jobs with `activeDeadlineSeconds` timeouts

All GPU jobs have `pipeline.wbc/phase` and `pipeline.wbc/component` pod labels for log aggregation.

## Key Design Decisions

- **Separate env config files per variant** rather than a single parameterized class. Matches Isaac Lab's own pattern (`G1FlatEnvCfg` vs `G1RoughEnvCfg`) and avoids fighting `@configclass` (attrs-based) inheritance semantics.
- **`JointPreset` dataclass** as single source of truth for joint ordering, defaults, action scale, and phase oscillator. Adding a preset doesn't require touching any config class.
- **`GridPatternCfg` for warehouse lidar** instead of `LidarPatternCfg`. Isaac Lab v2.3.2 doesn't expose angular lidar natively. The grid pattern produces a spatial sweep (meters), not angular (degrees). Documented in the scene config.
- **Phase oscillator** as a 4-dim gait clock `[cos(L), cos(R), sin(L), sin(R)]` with configurable frequency and offset. Returns zeros when disabled (`freq_hz=0`), preserving observation shape.
- **`EXPECTED_OBS_DIM` / `EXPECTED_ACTION_DIM`** class constants on each env config, validated at ONNX export time. Catches dim mismatches before training hours are wasted.
- **Multi-stage Containerfile** so source code changes rebuild in seconds instead of rebuilding the full Isaac Lab stack.

## Development

```bash
pip install -e ".[dev]"     # requires Python >=3.10, <3.13 (Isaac Sim constraint)
ruff check src/ tests/      # lint
ruff format src/ tests/     # format
pytest tests/ -v            # 93 tests (no GPU needed)
```

Tests validate joint presets, observation contracts, ONNX structure, phase oscillator behavior, and env registration. Tests that require Isaac Lab runtime are auto-skipped outside the container.
