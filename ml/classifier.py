"""Classifier training helpers."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier


def build_classifiers() -> dict[str, object]:
    """Return two classical ML classifiers for bonus comparison."""
    return {
        "multinomial_naive_bayes": MultinomialNB(alpha=0.5),
        "logistic_regression": OneVsRestClassifier(
            LogisticRegression(
                C=2.0,
                max_iter=1000,
                solver="liblinear",
                random_state=42,
            )
        ),
    }
