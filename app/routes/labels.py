"""Dataset label route."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.responses import LabelsResponse
from app.services.model_service import get_labels


router = APIRouter(tags=["labels"])


@router.get("/labels", response_model=LabelsResponse)
def labels() -> LabelsResponse:
    return LabelsResponse(labels=get_labels())
