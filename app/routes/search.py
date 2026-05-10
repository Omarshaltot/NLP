"""Search API route."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.requests import SearchRequest
from app.schemas.responses import SearchResponse
from app.services.search_service import search_documents


router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    results = search_documents(query=request.query, top_k=request.top_k)
    return SearchResponse(query=request.query, top_k=request.top_k, results=results)
