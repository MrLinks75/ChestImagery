from __future__ import annotations
import mlflow
from pathlib import Path
from src.utils.config import Config


def init_mlflow(experiment_name: str, tracking_uri: str = "sqlite:///mlflow.db") -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_config(cfg: Config) -> None:
    mlflow.log_params(vars(cfg))


def log_epoch_metrics(metrics: dict, step: int) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str | Path) -> None:
    mlflow.log_artifact(str(path))


def log_model_checkpoint(checkpoint_path: str | Path, artifact_dir: str = "checkpoints") -> None:
    mlflow.log_artifact(str(checkpoint_path), artifact_path=artifact_dir)
