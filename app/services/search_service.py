"""Search service based on TF-IDF vectors and cosine similarity."""

from __future__ import annotations

import re
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from app.services.artifact_loader import load_artifacts


def _make_snippet(document: str, max_length: int = 450) -> str:
    """Clean a document preview for API/UI display."""
    snippet = re.sub(r"\s+", " ", document).strip()
    if len(snippet) > max_length:
        return snippet[: max_length - 3] + "..."
    return snippet


def search_documents(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Return the top-k most similar documents for a user query."""
    artifacts = load_artifacts()
    vectorizer = artifacts["vectorizer"]
    search_index = artifacts["search_index"]
    class_names = artifacts["class_names"]

    query_vector = vectorizer.transform([query])
    document_vectors = search_index["document_vectors"]
    scores = cosine_similarity(query_vector, document_vectors).ravel()
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for rank, doc_index in enumerate(top_indices, start=1):
        target_index = int(search_index["targets"][doc_index])
        results.append(
            {
                "rank": rank,
                "document_id": int(doc_index),
                "similarity_score": float(scores[doc_index]),
                "predicted_label": class_names[target_index],
                "snippet": _make_snippet(search_index["documents"][doc_index]),
            }
        )
    return results
