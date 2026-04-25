-include .env
export

IMAGE ?= quay.io/jary/isaaclab-g1-train
SONIC_IMAGE ?= quay.io/jary/wbc-sonic
GALLERY_IMAGE ?= quay.io/jary/wbc-gallery
VLA_IMAGE ?= quay.io/jary/wbc-vla
TAG ?= latest
NAMESPACE ?= wbc-training

# Model Registry Postgres credentials (set in .env)
MODEL_REGISTRY_DB_USER ?= mlmduser
MODEL_REGISTRY_DB_NAME ?= wbc_model_registry

# ── Job registry ────────────────────────────────────────────────────
# Map JOB names to YAML files and K8s job names.
# Add new jobs here — no new targets needed.
#
#   make job-deploy JOB=test-flat
#   make job-logs JOB=test-flat
#   make job-clean JOB=test-flat

JOB_FILE_smoke-test              = deploy/jobs/smoke-test.yaml
JOB_NAME_smoke-test              = smoke-test
JOB_NEEDS_INFRA_smoke-test       = false

JOB_FILE_test-flat               = deploy/jobs/test-flat.yaml
JOB_NAME_test-flat               = test-flat
JOB_NEEDS_INFRA_test-flat        = false

JOB_FILE_test-rough              = deploy/jobs/test-rough.yaml
JOB_NAME_test-rough              = test-rough
JOB_NEEDS_INFRA_test-rough       = false

JOB_FILE_test-warehouse          = deploy/jobs/test-warehouse.yaml
JOB_NAME_test-warehouse          = test-warehouse
JOB_NEEDS_INFRA_test-warehouse   = false

JOB_FILE_test-preset             = deploy/jobs/test-preset.yaml
JOB_NAME_test-preset             = test-preset
JOB_NEEDS_INFRA_test-preset      = false

JOB_FILE_training-flat-6k        = deploy/jobs/training-flat-6k.yaml
JOB_NAME_training-flat-6k        = training-flat-6k
JOB_NEEDS_INFRA_training-flat-6k = true

JOB_FILE_training-rough-6k      = deploy/jobs/training-rough-6k.yaml
JOB_NAME_training-rough-6k      = training-rough-6k
JOB_NEEDS_INFRA_training-rough-6k = true

JOB_FILE_training-warehouse-6k  = deploy/jobs/training-warehouse-6k.yaml
JOB_NAME_training-warehouse-6k  = training-warehouse-6k
JOB_NEEDS_INFRA_training-warehouse-6k = true

JOB_FILE_training-preset-6k     = deploy/jobs/training-preset-6k.yaml
JOB_NAME_training-preset-6k     = training-preset-6k
JOB_NEEDS_INFRA_training-preset-6k = true

# Resolve JOB variable to file/name/infra-flag
_JOB_FILE       = $(JOB_FILE_$(JOB))
_JOB_NAME       = $(JOB_NAME_$(JOB))
_JOB_NEEDS_INFRA = $(JOB_NEEDS_INFRA_$(JOB))

.PHONY: build push ngc-login local-smoke-test \
        build-sonic push-sonic \
        build-gallery push-gallery \
        build-vla push-vla \
        deploy-infra deploy-pytorchjob-infra deploy-model-registry \
        job-deploy job-logs job-clean job-list \
        pipeline-compile pipeline-compile-distributed \
        sonic-pipeline-compile \
        vla-pipeline-compile \
        pipeline-deploy \
        download-checkpoint play record-video record-all-videos \
        lint test

# ── Container ────────────────────────────────────────────────────────
ngc-login:
	@echo "$(NGC_API_KEY)" | podman login nvcr.io -u '$$oauthtoken' --password-stdin

build:
	@if ! podman image exists $(IMAGE):$(TAG) 2>/dev/null; then $(MAKE) --no-print-directory ngc-login; fi
	podman build --format docker -t $(IMAGE):$(TAG) -f Containerfile .

push: build
	podman push $(IMAGE):$(TAG)

build-sonic:
	podman build --format docker -t $(SONIC_IMAGE):$(TAG) -f Containerfile.sonic .

push-sonic: build-sonic
	podman push $(SONIC_IMAGE):$(TAG)

build-gallery:
	podman build --format docker -t $(GALLERY_IMAGE):$(TAG) -f Containerfile.gallery .

push-gallery: build-gallery
	podman push $(GALLERY_IMAGE):$(TAG)

build-vla:
	podman build --format docker -t $(VLA_IMAGE):$(TAG) -f Containerfile.vla .

push-vla: build-vla
	podman push $(VLA_IMAGE):$(TAG)

# ── Local GPU smoke test (Podman + CDI) ──────────────────────────────
local-smoke-test:
	podman run --rm --device nvidia.com/gpu=all --env ACCEPT_EULA=Y --env PYTHONUNBUFFERED=1 \
		$(IMAGE):$(TAG) \
		-m wbc_pipeline.train \
		--task WBC-Velocity-Flat-G1-29DOF-v0 --headless --num_envs 64 --max_iterations 10

# ── OCP infrastructure ──────────────────────────────────────────────
deploy-infra:
	oc apply -f deploy/infra/namespace.yaml
	oc project $(NAMESPACE)
	oc apply -f deploy/infra/gpu-scc.yaml
	oc apply -f deploy/infra/minio.yaml
	oc apply -f deploy/infra/mlflow.yaml
	oc apply -f deploy/infra/mlflow-rbac.yaml
	oc delete job minio-init -n $(NAMESPACE) --ignore-not-found
	oc apply -f deploy/infra/minio-init.yaml
	oc wait --for=condition=complete job/minio-init -n $(NAMESPACE) --timeout=120s
	oc apply -f deploy/infra/dspa.yaml
	@echo "Waiting for DSPA to be ready..."
	oc wait --for=condition=Ready dspa/dspa -n $(NAMESPACE) --timeout=300s
	oc apply -f deploy/infra/dspa-rbac.yaml
	@echo "DSPA deployed. Verify runner SA: oc get sa -n $(NAMESPACE) | grep dspa"
ifdef HF_TOKEN
	oc create secret generic hf-credentials -n $(NAMESPACE) \
		--from-literal=HF_TOKEN=$(HF_TOKEN) \
		--dry-run=client -o yaml | oc apply -f -
	@echo "HF credentials secret created/updated."
endif
	oc apply -f deploy/infra/gallery.yaml
	@echo "Video gallery deployed. Route:"
	@oc get route video-gallery -n $(NAMESPACE) -o jsonpath='https://{.spec.host}{"\n"}' 2>/dev/null || echo "  (route pending)"

# ── Model Registry infrastructure ──────────────────────────────────
deploy-model-registry:
ifndef MODEL_REGISTRY_DB_PASSWORD
	$(error MODEL_REGISTRY_DB_PASSWORD is required. Set it in .env)
endif
	oc create secret generic wbc-model-registry-db -n rhoai-model-registries \
		--from-literal=POSTGRES_USER=$(MODEL_REGISTRY_DB_USER) \
		--from-literal=POSTGRES_PASSWORD=$(MODEL_REGISTRY_DB_PASSWORD) \
		--from-literal=POSTGRES_DB=$(MODEL_REGISTRY_DB_NAME) \
		--dry-run=client -o yaml | oc apply -f -
	oc apply -f deploy/infra/model-registry.yaml
	@echo "Model Registry deployed. Secret created from MODEL_REGISTRY_DB_* vars."

# ── PyTorchJob infrastructure ──────────────────────────────────────
deploy-pytorchjob-infra:
	oc apply -f deploy/infra/training-operator-rbac.yaml
	WBC_NAMESPACE=$(NAMESPACE) envsubst < deploy/infra/kueue.yaml | oc apply -f -
	@echo "PyTorchJob RBAC and Kueue resources deployed."

# ── Parametric job management ────────────────────────────────────────
#
# Usage:
#   make job-deploy JOB=training-flat-6k    # deploy infra (if needed) + create job
#   make job-logs JOB=training-flat-6k      # tail logs
#   make job-clean JOB=training-flat-6k     # delete job
#   make job-list                           # show available JOB names

job-deploy:
ifndef JOB
	$(error JOB is required. Run 'make job-list' to see available jobs)
endif
ifeq ($(_JOB_FILE),)
	$(error Unknown JOB '$(JOB)'. Run 'make job-list' to see available jobs)
endif
ifeq ($(_JOB_NEEDS_INFRA),true)
	@$(MAKE) --no-print-directory deploy-infra
else
	oc apply -f deploy/infra/namespace.yaml
	oc project $(NAMESPACE)
	oc apply -f deploy/infra/gpu-scc.yaml
endif
	oc delete job $(_JOB_NAME) -n $(NAMESPACE) --ignore-not-found
	oc apply -f $(_JOB_FILE)

job-logs:
ifndef JOB
	$(error JOB is required. Run 'make job-list' to see available jobs)
endif
ifeq ($(_JOB_NAME),)
	$(error Unknown JOB '$(JOB)'. Run 'make job-list' to see available jobs)
endif
	oc logs -f job/$(_JOB_NAME) -n $(NAMESPACE)

job-clean:
ifndef JOB
	$(error JOB is required. Run 'make job-list' to see available jobs)
endif
ifeq ($(_JOB_NAME),)
	$(error Unknown JOB '$(JOB)'. Run 'make job-list' to see available jobs)
endif
	oc delete job $(_JOB_NAME) -n $(NAMESPACE) --ignore-not-found

job-list:
	@echo "Available jobs (use with JOB=<name>):"
	@echo ""
	@echo "  Quick tests (10 iters, 64 envs):"
	@echo "    smoke-test           - container sanity check"
	@echo "    test-flat            - flat terrain, obs=103"
	@echo "    test-rough           - rough terrain, obs=290"
	@echo "    test-warehouse       - warehouse scene + lidar, obs=466"
	@echo "    test-preset          - Isaac Lab stock preset, action_scale=0.5"
	@echo ""
	@echo "  Training (6000 iters, 4096 envs + S3 + ONNX):"
	@echo "    training-flat-6k     - flat terrain"
	@echo "    training-rough-6k    - rough terrain"
	@echo "    training-warehouse-6k - warehouse scene"
	@echo "    training-preset-6k   - Isaac Lab stock preset"

# ── Pipeline ────────────────────────────────────────────────────────
pipeline-compile:
	python -m wbc_pipeline.pipeline

pipeline-compile-distributed:
	python -c "from kfp import compiler; from wbc_pipeline.pipeline import wbc_training_pytorchjob_pipeline; compiler.Compiler().compile(wbc_training_pytorchjob_pipeline, 'wbc_training_pytorchjob_pipeline.yaml'); print('Pipeline compiled to wbc_training_pytorchjob_pipeline.yaml')"

sonic-pipeline-compile:
	python -m wbc_pipeline.sonic.pipeline

vla-pipeline-compile:
	python -m wbc_pipeline.vla.pipeline

pipeline-deploy:
	oc apply -f deploy/infra/dspa.yaml
	@echo "Waiting for DSPA to be ready..."
	oc wait --for=condition=Ready dspa/dspa -n $(NAMESPACE) --timeout=300s
	oc apply -f deploy/infra/dspa-rbac.yaml
	@echo "DSPA deployed. Access pipeline UI via RHOAI dashboard."

# ── Local visualization ─────────────────────────────────────────────
#
# Usage:
#   make download-checkpoint                                          # download latest checkpoint from MinIO
#   make play                                                         # play latest checkpoint in GUI
#   make play TASK=WBC-Velocity-Rough-G1-29DOF-v0                     # play with a different env
#   make record-video TASK=WBC-Velocity-Rough-G1-29DOF-v0             # record video with a different env

TASK ?= WBC-Velocity-Flat-G1-29DOF-v0

# Extract terrain keyword from TASK for checkpoint filtering (e.g. Flat → *flat*)
_TASK_TERRAIN = $(shell echo '$(TASK)' | sed -n 's/.*Velocity-\([A-Za-z]*\)-.*/\1/p' | tr A-Z a-z)

download-checkpoint:
	python3 -m wbc_pipeline.download_checkpoint

play:
	@CKPT=$$(find checkpoints -name '*$(_TASK_TERRAIN)*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-); \
	if [ -z "$$CKPT" ]; then \
		echo "No checkpoints matching '$(_TASK_TERRAIN)' found. Available:"; \
		ls checkpoints/*.pt 2>/dev/null || echo "  (none)"; \
		exit 1; \
	fi; \
	echo "Playing: $$CKPT (task: $(TASK))"; \
	podman run --rm --device nvidia.com/gpu=all \
		-e ACCEPT_EULA=Y -e DISPLAY=$$DISPLAY \
		-e XAUTHORITY=/tmp/.Xauthority \
		-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
		-v $${XAUTHORITY:-$$HOME/.Xauthority}:/tmp/.Xauthority:ro \
		--network=host --ipc=host \
		-v $$(pwd)/checkpoints:/checkpoints:ro \
		-v $$(pwd)/src:/workspace/wbc_pipeline/src:ro \
		$(IMAGE):$(TAG) \
		-m wbc_pipeline.play \
			--task $(TASK) \
			--checkpoint /checkpoints/$$(basename $$CKPT) \
			--num_envs 4

record-video:
	@CKPT=$$(find checkpoints -name '*$(_TASK_TERRAIN)*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-); \
	if [ -z "$$CKPT" ]; then \
		echo "No checkpoints matching '$(_TASK_TERRAIN)' found. Available:"; \
		ls checkpoints/*.pt 2>/dev/null || echo "  (none)"; \
		exit 1; \
	fi; \
	mkdir -p videos; \
	echo "Recording: $$CKPT (task: $(TASK))"; \
	podman run --rm --device nvidia.com/gpu=all \
		-e ACCEPT_EULA=Y \
		--ipc=host \
		-v $$(pwd)/checkpoints:/checkpoints:ro \
		-v $$(pwd)/src:/workspace/wbc_pipeline/src:ro \
		-v $$(pwd)/videos:/videos \
		$(IMAGE):$(TAG) \
		-m wbc_pipeline.play \
			--task $(TASK) \
			--checkpoint /checkpoints/$$(basename $$CKPT) \
			--num_envs 4 \
			--headless --enable_cameras --video \
			--video_dir /videos \
			--video_length 500 \
			--max_steps 500

record-all-videos:
	@CKPTS=$$(find checkpoints -name '*$(_TASK_TERRAIN)*.pt' 2>/dev/null | head -1); \
	if [ -z "$$CKPTS" ]; then \
		echo "No checkpoints matching '$(_TASK_TERRAIN)' found. Available:"; \
		ls checkpoints/*.pt 2>/dev/null || echo "  (none)"; \
		exit 1; \
	fi; \
	mkdir -p videos; \
	echo "Recording videos from checkpoints/..."; \
	podman run --rm --device nvidia.com/gpu=all \
		-e ACCEPT_EULA=Y \
		--ipc=host \
		-v $$(pwd)/checkpoints:/checkpoints:ro \
		-v $$(pwd)/src:/workspace/wbc_pipeline/src:ro \
		-v $$(pwd)/videos:/videos \
		$(IMAGE):$(TAG) \
		-m wbc_pipeline.record_training_videos \
			--task $(TASK) \
			--checkpoint_dir /checkpoints \
			--video_dir /videos \
			--steps 500

# ── Development ──────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

test:
	pytest tests/ -v
