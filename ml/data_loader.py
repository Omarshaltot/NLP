"""Dataset loading utilities for the 20 Newsgroups dataset."""

from __future__ import annotations

from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch


def load_20newsgroups_data() -> tuple[Bunch, Bunch]:
    """Load train and test splits from scikit-learn's 20 Newsgroups dataset."""
    remove_parts = ("headers", "footers", "quotes")
    train_data = fetch_20newsgroups(
        subset="train",
        remove=remove_parts,
        shuffle=True,
        random_state=42,
    )
    test_data = fetch_20newsgroups(
        subset="test",
        remove=remove_parts,
        shuffle=True,
        random_state=42,
    )
    return train_data, test_data
