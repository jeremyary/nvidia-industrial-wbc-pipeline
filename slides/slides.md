---
theme: default
title: "Red Hat Contributions: Embodied AI Training Pipeline"
info: |
  Red Hat's contributions to the humanoid robotics training infrastructure
  for the Unitree G1 — NVIDIA collaboration.
---

# Red Hat + NVIDIA: Teaching a Humanoid to Move

ML training infrastructure for the Unitree G1 — built on OpenShift AI

<br>

<div class="grid grid-cols-2 gap-8">
<div>

### The Problem
A humanoid robot needs three models working in concert: a mission planner (Nemotron), a vision-language-action model (GR00T N1.7), and a whole-body controller (GEAR-SONIC). Preparing these models for deployment requires GPU-accelerated pipelines with model validation, versioning, and multi-tenant scheduling.

### What Red Hat Built
KFP v2 pipelines on RHOAI that fine-tune GR00T N1.7-3B on real G1 teleoperation data, validate exported ONNX models, and register them — parameterized for different datasets and training configurations. A separate pipeline imports and validates the pre-trained GEAR-SONIC controller.

</div>
<div>

### Key Infrastructure
- **DSPA** orchestrates multi-step GPU pipelines via Argo Workflows
- **MinIO** caches HuggingFace models and datasets in S3
- **MLflow** deployed for experiment tracking (WBC pipeline integration)
- **Model Registry** versions ONNX artifacts with S3 URIs and metadata
- **Kueue** GPU quota management for multi-tenant clusters

### Impact
Swap training datasets at submission time, adjust steps and batch size — same pipeline, different robot skills.

</div>
</div>

---

# Training the Robot Brain on OpenShift AI

Red Hat's ML pipeline infrastructure for the Unitree G1 humanoid

<br>

Three-tier control stack — Red Hat built the training and validation pipelines:

```
Mission Planner (Nemotron)  →  VLA (GR00T N1.7)  →  WBC (GEAR-SONIC)
  "go to the loading dock"     camera + language →     motion commands →
                               motion commands          joint torques
```

<br>

- **VLA fine-tuning** — GR00T N1.7-3B on real G1 teleoperation data (GPU training on L40S)
- **WBC import** — GEAR-SONIC pre-trained whole-body controller (validated + registered on-cluster)

---

# Pipeline Architecture on RHOAI

KFP v2 pipelines on Data Science Pipelines Application (DSPA)

<div class="grid grid-cols-2 gap-8">
<div>

### VLA Fine-Tuning Pipeline
```
data_prep (CPU)
  Download base model + dataset
  from HuggingFace, cache in S3
       ↓
fine_tune_and_export (GPU)
  torchrun GR00T fine-tuning
  ONNX export (6 components)
       ↓
validate_onnx (CPU)
  Structure, inference, determinism
       ↓
register_model (CPU)
  RHOAI Model Registry
```

</div>
<div>

### Cluster Infrastructure
- **MinIO** — S3 storage for checkpoints, models, datasets
- **MLflow** — Experiment tracking server
- **DSPA** — KFP v2 orchestration via Argo Workflows
- **Model Registry** — ONNX versioning with S3 URIs
- **Kueue** — GPU quota management

### GPU Target
- **L40S** (48GB) — VLA fine-tuning

</div>
</div>

---

# What Red Hat Built

<div class="grid grid-cols-2 gap-8">
<div>

### Training Infrastructure
- **KFP v2 pipelines** on RHOAI for VLA fine-tuning and SONIC model import
- **Purpose-built containers** — CUDA 12.8 + GR00T N1.7 (VLA), Python slim (SONIC)
- **Custom modality config** mapping real G1 teleoperation data to GR00T's training framework

### Model Lifecycle
- **Automated ONNX validation** — structural check, shape correctness, inference, determinism
- **Model Registry integration** — versioned models with S3 URIs and metadata
- **S3 dataset caching** — HuggingFace downloads cached in MinIO, skip on re-run

</div>
<div>

### OpenShift Integration
- **Kueue** GPU quota management for multi-tenant scheduling
- **DSPA RBAC** — SA-based auth for MLflow + Model Registry
- **Video gallery** — training progress videos served from S3

### Parameterized Scenarios
- Swap training datasets at submission time (HuggingFace repo ID)
- Adjust steps and batch size per pipeline run
- Same pipeline, different robot behavior

</div>
</div>

---

# Red Hat + NVIDIA: Teaching a Humanoid to Move

ML training infrastructure for the Unitree G1 — built on OpenShift AI

<!-- 5-slide version starts here — one section per slide, room for graphics -->

<!-- GRAPHIC: Hero image of the Unitree G1 humanoid robot in an industrial setting (warehouse or factory floor). Clean, professional, dark background with subtle red accent lighting. The robot should be mid-stride or standing ready, conveying capability and precision. -->

---

# The Problem

A humanoid robot needs three models working in concert

<!-- GRAPHIC: Three-tier control stack diagram flowing left to right. Three distinct blocks connected by arrows: (1) "Mission Planner / Nemotron" with a speech bubble icon showing "go to the loading dock", (2) "VLA / GR00T N1.7" with a camera + text icon showing camera frames and language going in, motion commands coming out, (3) "WBC / GEAR-SONIC" with a robot joint icon showing motion commands going in, joint torques coming out. Clean flat design, dark charcoal background, white text, red arrows between stages. -->

- **Nemotron** interprets high-level commands into navigation goals
- **GR00T N1.7** translates camera input + language into motion commands
- **GEAR-SONIC** converts motion commands into 29-DOF joint torques

Each model must be prepared separately — fine-tuned, validated, and versioned — before the stack works end-to-end.

---

# What Red Hat Built

KFP v2 pipelines on RHOAI for model preparation and validation

<!-- GRAPHIC: Pipeline flow diagram showing two pipelines stacked vertically. Top: "VLA Fine-Tuning Pipeline" with four connected stages as rounded rectangles flowing left to right — data_prep (CPU, blue) → fine_tune_and_export (GPU, red) → validate_onnx (CPU, blue) → register_model (CPU, blue). Each box has a one-line subtitle: "HuggingFace → S3 cache", "torchrun + ONNX export", "structure/shape/inference/determinism", "Model Registry". Bottom: "SONIC Import Pipeline" with three stages — fetch_checkpoint → validate_onnx → register_model. Dark background, flat design, GPU stage visually emphasized with red fill. -->

### GEAR-SONIC Import Pipeline
Fetch pre-trained checkpoint from HuggingFace → validate ONNX → register in Model Registry

---

# Cluster Infrastructure

Platform services deployed on OpenShift AI

<!-- GRAPHIC: Cluster architecture diagram. A large rounded rectangle labeled "OpenShift AI Cluster" containing six service blocks arranged in a grid or ring: DSPA (Argo Workflows icon), MinIO (storage bucket icon), MLflow (chart icon), Model Registry (package/version icon), Kueue (queue icon), and an L40S GPU node. Arrows show data flow: pipelines (DSPA) read/write to MinIO, log to MLflow, register to Model Registry, and schedule GPU work through Kueue. Dark background, service blocks in charcoal with white text, red border on the outer cluster rectangle. -->

- **DSPA** — KFP v2 orchestration via Argo Workflows
- **MinIO** — S3 storage for model/dataset caching, checkpoints, ONNX artifacts
- **MLflow** — Experiment tracking server
- **Model Registry** — ONNX versioning with S3 URIs and metadata
- **Kueue** — GPU quota management for multi-tenant scheduling

### GPU Target
- **L40S** (48GB) — VLA fine-tuning with custom modality config for real G1 teleoperation data

---

# Impact

Same pipeline, different robot skills

<!-- GRAPHIC: Side-by-side comparison showing the same pipeline used with two different configurations. Left side: "Venue A — Pick and Place" with a dataset card showing "nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1" and parameters "steps: 2000, batch: 64". Right side: "Venue B — Pouring" with a dataset card showing "unitreerobotics/G1_Pouring_Dataset" and parameters "steps: 5000, batch: 32". A single pipeline graphic spans both, showing that only the inputs change. Clean flat design, dark background, red highlights on the parameter values that differ. -->

### Parameterized for Any Venue
- Swap training datasets at submission time — pass a HuggingFace repo ID
- Adjust training steps and batch size per pipeline run
- Custom modality config maps real G1 teleoperation data to GR00T's training framework

### Purpose-Built Containers
- **VLA** — CUDA 12.8 + GR00T N1.7, torchcodec for video frame decoding
- **SONIC** — Python slim, lightweight ONNX validation and registration

### Training Visibility
- Video gallery serves training progress recordings from S3
- Automated ONNX validation gates model registration
