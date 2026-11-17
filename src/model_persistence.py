"""Save and load fitted model pipelines with Joblib."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


DEFAULT_MODEL_DIR = Path("models")
MODEL_NAMES = {"logistic_regression", "random_forest"}


def save_model_pipelines(
    models: dict[str, Pipeline], metadata: dict[str, Any], model_dir: str | Path = DEFAULT_MODEL_DIR
) -> dict[str, Path]:
    """Persist full fitted pipelines and companion JSON metadata."""
    destination = Path(model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, Path] = {}
    for name, pipeline in models.items():
        if name not in MODEL_NAMES:
            raise ValueError(f"Unsupported model name: {name}")
        artifact = destination / f"{name}_pipeline.joblib"
        joblib.dump(pipeline, artifact, compress=3)
        artifact_paths[name] = artifact
    metadata_path = destination / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    artifact_paths["metadata"] = metadata_path
    return artifact_paths


def load_model_pipeline(model_name: str, model_dir: str | Path = DEFAULT_MODEL_DIR) -> Pipeline:
    """Load one saved pipeline, including its fitted preprocessing steps."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown saved model: {model_name}")
    path = Path(model_dir) / f"{model_name}_pipeline.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Saved pipeline not found: {path}")
    return joblib.load(path)
