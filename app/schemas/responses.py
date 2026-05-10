"""Response schemas for the FastAPI app."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class SearchResult(BaseModel):
    rank: int
    document_id: int
    similarity_score: float
    predicted_label: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchResult]


class PredictResponse(BaseModel):
    class_index: int
    class_label: str


class LabelsResponse(BaseModel):
    labels: list[str]


class MetricsResponse(BaseModel):
    metrics: dict[str, Any]
