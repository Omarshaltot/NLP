"""Train the search index and classifiers, then save all project artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from ml.classifier import build_classifiers
from ml.data_loader import load_20newsgroups_data
from ml.evaluation import evaluate_classifier, save_confusion_matrix
from ml.vectorizer import build_tfidf_vectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def train_and_save(max_features: int = 30000) -> dict[str, Any]:
    """Run the complete training pipeline and persist artifacts with joblib."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    print("Loading 20 Newsgroups dataset...")
    train_data, test_data = load_20newsgroups_data()
    class_names = list(train_data.target_names)

    print("Fitting TF-IDF vectorizer...")
    vectorizer = build_tfidf_vectorizer(max_features=max_features)
    x_train = vectorizer.fit_transform(train_data.data)
    x_test = vectorizer.transform(test_data.data)

    print("Training and comparing classifiers...")
    model_results: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, object] = {}

    for model_name, classifier in build_classifiers().items():
        print(f"Training {model_name}...")
        classifier.fit(x_train, train_data.target)
        predictions = classifier.predict(x_test)
        metrics = evaluate_classifier(
            model_name=model_name,
            y_true=test_data.target,
            y_pred=predictions,
            class_names=class_names,
        )
        model_results[model_name] = metrics
        trained_models[model_name] = classifier
        print(
            f"{model_name}: accuracy={metrics['accuracy']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )

    best_model_name = max(
        model_results,
        key=lambda name: model_results[name]["f1_macro"],
    )
    best_model = trained_models[best_model_name]
    best_predictions = best_model.predict(x_test)

    print(f"Best model: {best_model_name}")
    save_confusion_matrix(
        y_true=test_data.target,
        y_pred=best_predictions,
        class_names=class_names,
        output_path=ARTIFACTS_DIR / "confusion_matrix.png",
    )

    metrics_payload = {
        "dataset": "20 Newsgroups from scikit-learn",
        "selected_model": best_model_name,
        "model_comparison": model_results,
        "train_size": len(train_data.data),
        "test_size": len(test_data.data),
        "number_of_classes": len(class_names),
        "class_names": class_names,
        "vectorizer": {
            "type": "TfidfVectorizer",
            "max_features": max_features,
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.9,
            "sublinear_tf": True,
        },
    }

    print("Saving artifacts...")
    joblib.dump(best_model, ARTIFACTS_DIR / "classifier.joblib")
    joblib.dump(vectorizer, ARTIFACTS_DIR / "vectorizer.joblib")
    joblib.dump(class_names, ARTIFACTS_DIR / "class_names.joblib")
    joblib.dump(
        {
            "documents": train_data.data,
            "targets": train_data.target,
            "document_vectors": x_train,
            "class_names": class_names,
        },
        ARTIFACTS_DIR / "search_index.joblib",
    )

    with (ARTIFACTS_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    print(f"Artifacts saved in: {ARTIFACTS_DIR}")
    return metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 20 Newsgroups project.")
    parser.add_argument(
        "--max-features",
        type=int,
        default=30000,
        help="Maximum number of TF-IDF features to keep.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save(max_features=args.max_features)
