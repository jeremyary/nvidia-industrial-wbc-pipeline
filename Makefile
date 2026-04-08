-include .env
export

IMAGE ?= quay.io/jary/isaaclab-g1-train
TAG ?= latest
NAMESPACE ?= wbc-training

.PHONY: build push ngc-login smoke-test deploy-namespace deploy-smoke-test \
        logs-smoke-test clean-smoke-test deploy-phase1-validation \
        logs-phase1-validation clean-phase1-validation \
        deploy-phase3b-validation logs-phase3b-flat logs-phase3b-rough \
        logs-phase3b-isaaclab clean-phase3b-validation lint test

# ── Container ────────────────────────────────────────────────────────
ngc-login:
	@echo "$(NGC_API_KEY)" | podman login nvcr.io -u '$$oauthtoken' --password-stdin

build: ngc-login
	podman build --format docker -t $(IMAGE):$(TAG) -f Containerfile .

push: build
	podman push $(IMAGE):$(TAG)

# ── Local GPU smoke test (Podman + CDI) ──────────────────────────────
smoke-test:
	podman run --rm --device nvidia.com/gpu=all --env ACCEPT_EULA=Y --env PYTHONUNBUFFERED=1 \
		$(IMAGE):$(TAG) \
		-m wbc_pipeline.train \
		--task WBC-Velocity-Flat-G1-29DOF-v0 --headless --num_envs 64 --max_iterations 10

# ── OCP cluster deployment ───────────────────────────────────────────
deploy-namespace:
	oc apply -f deploy/namespace.yaml
	oc project $(NAMESPACE)

deploy-smoke-test: deploy-namespace
	oc apply -f deploy/gpu-scc.yaml
	oc apply -f deploy/smoke-test-job.yaml

logs-smoke-test:
	oc logs -f job/isaaclab-smoke-test -n $(NAMESPACE)

clean-smoke-test:
	oc delete job isaaclab-smoke-test -n $(NAMESPACE) --ignore-not-found

deploy-phase1-validation: deploy-namespace
	oc apply -f deploy/gpu-scc.yaml
	oc delete job phase1-validation -n $(NAMESPACE) --ignore-not-found
	oc apply -f deploy/phase1-validation-job.yaml

logs-phase1-validation:
	oc logs -f job/phase1-validation -n $(NAMESPACE)

clean-phase1-validation:
	oc delete job phase1-validation -n $(NAMESPACE) --ignore-not-found

deploy-phase3b-validation: deploy-namespace
	oc apply -f deploy/gpu-scc.yaml
	oc delete job phase3b-flat-regression phase3b-rough-validation phase3b-isaaclab-preset -n $(NAMESPACE) --ignore-not-found
	oc apply -f deploy/phase3b-validation-jobs.yaml

logs-phase3b-flat:
	oc logs -f job/phase3b-flat-regression -n $(NAMESPACE)

logs-phase3b-rough:
	oc logs -f job/phase3b-rough-validation -n $(NAMESPACE)

logs-phase3b-isaaclab:
	oc logs -f job/phase3b-isaaclab-preset -n $(NAMESPACE)

clean-phase3b-validation:
	oc delete job phase3b-flat-regression phase3b-rough-validation phase3b-isaaclab-preset -n $(NAMESPACE) --ignore-not-found

# ── Development ──────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

test:
	pytest tests/ -v
