"""Prediction API route."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.requests import PredictRequest
from app.schemas.responses import PredictResponse
from app.services.model_service import predict_category


router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    prediction = predict_category(request.text)
    return PredictResponse(**prediction)
