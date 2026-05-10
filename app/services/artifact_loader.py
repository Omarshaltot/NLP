"""Centralized loading of saved ML artifacts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when the training artifacts are missing."""


def _require_artifact(path: Path) -> Path:
    if not path.exists():
        raise ArtifactNotFoundError(
            f"Missing artifact: {path}. Run `python -m ml.train` first."
        )
    return path


@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, Any]:
    """Load all persisted artifacts once and reuse them across requests."""
    vectorizer = joblib.load(_require_artifact(ARTIFACTS_DIR / "vectorizer.joblib"))
    classifier = joblib.load(_require_artifact(ARTIFACTS_DIR / "classifier.joblib"))
    class_names = joblib.load(_require_artifact(ARTIFACTS_DIR / "class_names.joblib"))
    search_index = joblib.load(_require_artifact(ARTIFACTS_DIR / "search_index.joblib"))

    metrics_path = _require_artifact(ARTIFACTS_DIR / "metrics.json")
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    return {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "class_names": class_names,
        "search_index": search_index,
        "metrics": metrics,
    }
