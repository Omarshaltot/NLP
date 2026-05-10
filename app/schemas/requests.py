"""Request schemas for the FastAPI app."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to classify")
