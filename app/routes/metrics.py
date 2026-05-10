"""Evaluation metrics route."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.responses import MetricsResponse
from app.services.model_service import get_metrics


router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(metrics=get_metrics())
