"""Text preprocessing utilities shared by training, search, and prediction."""

from __future__ import annotations

import re
from typing import List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


TOKEN_PATTERN = re.compile(r"[a-z]{2,}")


def tokenize(text: str) -> List[str]:
    """Lowercase, remove punctuation/numbers, tokenize, and remove stop words."""
    text = text.lower()
    tokens = TOKEN_PATTERN.findall(text)
    return [token for token in tokens if token not in ENGLISH_STOP_WORDS]


def preprocess_text(text: str) -> str:
    """Return a cleaned whitespace-separated string for vectorizers."""
    return " ".join(tokenize(text))
