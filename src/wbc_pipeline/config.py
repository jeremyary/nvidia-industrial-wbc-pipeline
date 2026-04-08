# This project was developed with assistance from AI tools.
"""Training configuration via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class S3Config:
    """S3-compatible storage for checkpoints."""

    endpoint: str = field(default_factory=lambda: os.environ.get("S3_ENDPOINT", ""))
    bucket: str = field(default_factory=lambda: os.environ.get("S3_BUCKET", "wbc-training"))
    prefix: str = field(default_factory=lambda: os.environ.get("S3_PREFIX", "checkpoints"))
    access_key: str = field(default_factory=lambda: os.environ.get("AWS_ACCESS_KEY_ID", ""))
    secret_key: str = field(default_factory=lambda: os.environ.get("AWS_SECRET_ACCESS_KEY", ""))

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint and self.access_key and self.secret_key)


@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""

    tracking_uri: str = field(default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", ""))
    experiment_name: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_EXPERIMENT_NAME", "g1-29dof-locomotion")
    )
    insecure_tls: bool = field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "false").lower() == "true"
    )

    @property
    def enabled(self) -> bool:
        return bool(self.tracking_uri)


@dataclass
class TrainingConfig:
    """Top-level training configuration."""

    s3: S3Config = field(default_factory=S3Config)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    checkpoint_interval: int = field(default_factory=lambda: int(os.environ.get("CHECKPOINT_INTERVAL", "50")))
    resume_checkpoint: str = field(default_factory=lambda: os.environ.get("RESUME_CHECKPOINT", ""))
