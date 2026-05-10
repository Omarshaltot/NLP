"""Classification, label, and metrics services."""

from __future__ import annotations

from typing import Any

from app.services.artifact_loader import load_artifacts


def predict_category(text: str) -> dict[str, Any]:
    """Classify raw text using the saved vectorizer and classifier."""
    artifacts = load_artifacts()
    vectorizer = artifacts["vectorizer"]
    classifier = artifacts["classifier"]
    class_names = artifacts["class_names"]

    features = vectorizer.transform([text])
    class_index = int(classifier.predict(features)[0])
    return {
        "class_index": class_index,
        "class_label": class_names[class_index],
    }


def get_labels() -> list[str]:
    """Return the dataset class names."""
    return list(load_artifacts()["class_names"])


def get_metrics() -> dict[str, Any]:
    """Return saved evaluation metrics."""
    return load_artifacts()["metrics"]
