"""TF-IDF feature extraction for search and classification."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from ml.preprocessing import preprocess_text


def build_tfidf_vectorizer(max_features: int = 30000) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer using the project's preprocessing function."""
    return TfidfVectorizer(
        preprocessor=preprocess_text,
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )
