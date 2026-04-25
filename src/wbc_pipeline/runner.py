# This project was developed with assistance from AI tools.
"""Custom OnPolicyRunner with MLflow logging, S3 checkpointing, and SIGTERM handling."""

from __future__ import annotations

import os
import signal
import sys
import tempfile
from typing import TYPE_CHECKING

from rsl_rl.runners import OnPolicyRunner

if TYPE_CHECKING:
    from wbc_pipeline.config import TrainingConfig


class _MlflowTensorboardWriter:
    """Wraps a TensorBoard SummaryWriter to also log metrics to MLflow."""

    def __init__(self, tb_writer, experiment_name: str, tracking_uri: str, insecure_tls: bool = False):
        if insecure_tls:
            os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

        # Inject SA token for Kubernetes-auth MLflow servers (RHOAI)
        sa_token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        if os.path.exists(sa_token_path):
            with open(sa_token_path) as f:
                os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()

        import mlflow

        # Register workspace header provider for RHOAI MLflow
        sa_ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
        if os.path.exists(sa_ns_path):
            with open(sa_ns_path) as f:
                workspace = f.read().strip()
            from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider
            from mlflow.tracking.request_header.registry import _request_header_provider_registry

            class _WorkspaceHeaderProvider(RequestHeaderProvider):
                def in_context(self):
                    return True

                def request_headers(self):
                    return {"X-MLFLOW-WORKSPACE": workspace}

            _request_header_provider_registry.register(_WorkspaceHeaderProvider)

        self._tb = tb_writer
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run()
        self._mlflow = mlflow

    def add_scalar(self, tag, scalar_value, global_step=None, **kwargs):
        self._tb.add_scalar(tag, scalar_value, global_step, **kwargs)
        step = int(global_step) if global_step is not None else 0
        try:
            self._mlflow.log_metric(tag.replace("/", "_"), float(scalar_value), step=step)
        except Exception:
            pass

    def stop(self):
        if hasattr(self._tb, "stop"):
            self._tb.stop()
        self._mlflow.end_run()

    def __getattr__(self, name):
        return getattr(self._tb, name)


class WBCRunner(OnPolicyRunner):
    """OnPolicyRunner with MLflow metrics, S3 checkpoint upload, and SIGTERM handling."""

    def __init__(self, env, cfg: dict, log_dir: str, device: str, training_cfg: TrainingConfig, render_env=None):
        super().__init__(env, cfg, log_dir=log_dir, device=device)
        self._training_cfg = training_cfg
        self._s3_client = None
        self._shutting_down = False
        self._render_env = render_env

        if training_cfg.s3.enabled:
            self._init_s3()

        if training_cfg.checkpoint_interval:
            self.save_interval = training_cfg.checkpoint_interval

        self._video_iterations: set[int] = set()
        if training_cfg.video.enabled and render_env is not None:
            self._compute_video_iterations(cfg.get("max_iterations", 0))
            if self._video_iterations:
                print(f"[WBCRunner] Video recording at iterations: {sorted(self._video_iterations)}")

        signal.signal(signal.SIGTERM, self._sigterm_handler)

    def _compute_video_iterations(self, max_iters: int):
        """Pick ~N evenly-spaced iterations aligned to save_interval."""
        if max_iters <= 0:
            return
        n = self._training_cfg.video.num_recordings
        save_iv = self.save_interval
        total_saves = max_iters // save_iv
        if total_saves == 0:
            return
        step = max(1, total_saves // n)
        for idx in range(step, total_saves + 1, step):
            self._video_iterations.add(idx * save_iv)
        self._video_iterations.add(total_saves * save_iv)

    def _init_s3(self):
        s3_cfg = self._training_cfg.s3
        self._s3_client = s3_cfg.create_client()
        print(f"[WBCRunner] S3 checkpointing enabled: s3://{s3_cfg.bucket}/{s3_cfg.prefix}")

    def _prepare_logging_writer(self):
        """Create tensorboard writer, then wrap with MLflow if configured."""
        super()._prepare_logging_writer()
        if self._training_cfg.mlflow.enabled:
            mlflow_cfg = self._training_cfg.mlflow
            try:
                self.writer = _MlflowTensorboardWriter(
                    self.writer,
                    experiment_name=mlflow_cfg.experiment_name,
                    tracking_uri=mlflow_cfg.tracking_uri,
                    insecure_tls=mlflow_cfg.insecure_tls,
                )
                print(f"[WBCRunner] MLflow tracking enabled: {mlflow_cfg.tracking_uri}")
            except Exception as e:
                print(f"[WBCRunner] WARNING: MLflow init failed, continuing without it: {e}")

    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)
        if self._s3_client is not None:
            self._upload_to_s3(path)
        if self.current_learning_iteration in self._video_iterations:
            self._record_video()

    def _upload_to_s3(self, local_path: str):
        s3_cfg = self._training_cfg.s3
        filename = os.path.basename(local_path)
        s3_key = f"{s3_cfg.prefix}/{filename}"
        try:
            self._s3_client.upload_file(local_path, s3_cfg.bucket, s3_key)
            print(f"[WBCRunner] Uploaded checkpoint to s3://{s3_cfg.bucket}/{s3_key}")
        except Exception as e:
            print(f"[WBCRunner] WARNING: S3 upload failed for {filename}: {e}", file=sys.stderr)

    def _record_video(self):
        """Capture a short inference video at the current training checkpoint."""
        import imageio
        import numpy as np
        import torch

        iter_num = self.current_learning_iteration
        run_name = self._training_cfg.s3.prefix.replace("/", "_")
        video_name = f"{run_name}_iter_{iter_num:06d}.mp4"

        print(f"[WBCRunner] Recording video at iteration {iter_num}...")

        policy = self.get_inference_policy(device=self.device)
        obs = self.env.get_observations()
        frames = []
        with torch.inference_mode():
            for _ in range(self._training_cfg.video.steps_per_video):
                actions = policy(obs)
                obs, _, _, _ = self.env.step(actions)
                frame = self._render_env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))

        # Restore training mode after inference
        self.alg.policy.train()

        if not frames:
            print(f"[WBCRunner] WARNING: No frames captured at iteration {iter_num}")
            return

        video_dir = os.path.join(self.log_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, video_name)
        imageio.mimwrite(video_path, frames, fps=50, quality=8)
        print(f"[WBCRunner] Saved video: {video_path} ({len(frames)} frames)")

        if self._s3_client is not None:
            s3_key = f"{self._training_cfg.s3.prefix}/videos/{video_name}"
            try:
                self._s3_client.upload_file(video_path, self._training_cfg.s3.bucket, s3_key)
                print(f"[WBCRunner] Uploaded video to s3://{self._training_cfg.s3.bucket}/{s3_key}")
            except Exception as e:
                print(f"[WBCRunner] WARNING: S3 video upload failed: {e}", file=sys.stderr)

    def _sigterm_handler(self, signum, frame):
        if self._shutting_down:
            return
        self._shutting_down = True
        print("[WBCRunner] SIGTERM received — saving final checkpoint...")
        path = os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
        self.save(path)
        if isinstance(self.writer, _MlflowTensorboardWriter):
            self.writer.stop()
        print("[WBCRunner] Graceful shutdown complete.")
        sys.exit(143)

    def resume_from_s3(self, s3_key: str) -> dict:
        """Download a checkpoint from S3 and load it."""
        s3_cfg = self._training_cfg.s3
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            local_path = f.name
        try:
            self._s3_client.download_file(s3_cfg.bucket, s3_key, local_path)
            print(f"[WBCRunner] Downloaded checkpoint from s3://{s3_cfg.bucket}/{s3_key}")
            return self.load(local_path)
        finally:
            os.unlink(local_path)

    def resume_latest_from_s3(self) -> dict | None:
        """Find and load the latest checkpoint from S3."""
        if self._s3_client is None:
            return None
        s3_cfg = self._training_cfg.s3
        try:
            model_files = []
            paginator = self._s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=s3_cfg.bucket, Prefix=s3_cfg.prefix + "/"):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(".pt"):
                        model_files.append(obj)
            if not model_files:
                print("[WBCRunner] No .pt checkpoints found in S3.")
                return None
            latest = max(model_files, key=lambda obj: obj["LastModified"])
            return self.resume_from_s3(latest["Key"])
        except Exception as e:
            print(f"[WBCRunner] WARNING: Failed to list S3 checkpoints: {e}", file=sys.stderr)
            return None
