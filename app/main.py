"""FastAPI entry point for the 20 Newsgroups search engine project."""

from __future__ import annotations

from fastapi import FastAPI

from app.routes import health, labels, metrics, predict, search


app = FastAPI(
    title="Simple Search Engine API",
    description="TF-IDF document retrieval and text classification for 20 Newsgroups.",
    version="1.0.0",
)

app.include_router(health.router)
app.include_router(search.router)
app.include_router(predict.router)
app.include_router(labels.router)
app.include_router(metrics.router)
