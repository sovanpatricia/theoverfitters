"""
main.py
-------
FastAPI entry point for Music Mood Matcher.

Run with:
    uvicorn main:app --reload

Endpoints:
    POST /recommend   — main recommendation endpoint
    GET  /health      — health check
    GET  /            — API info
"""

import os
import logging
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.toxicity      import toxicity_filter
from src.rag           import generate_recommendations
from src.hallucination import HallucinationGuard

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)

BALANCED_PATH  = os.path.join("data", "balanced.csv")
PROCESSED_PATH = os.path.join("data", "processed.csv")


# ─── App state ────────────────────────────────────────────────────────────────

guard: HallucinationGuard | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load dataset and build hallucination guard on startup."""
    global guard
    path = BALANCED_PATH if os.path.exists(BALANCED_PATH) else PROCESSED_PATH
    if os.path.exists(path):
        df    = pd.read_csv(path)
        guard = HallucinationGuard(df)
        log.info(f"Loaded dataset: {len(df):,} tracks")
    else:
        log.warning("Dataset not found — hallucination guard disabled")
    yield


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Music Mood Matcher",
    description = "Describe your mood, get songs that match — powered by RAG + local SLM",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    mood: str = Field(
        ...,
        min_length = 3,
        max_length = 500,
        description = "Natural language mood description",
        examples    = ["I feel melancholic and nostalgic, like looking at old photos"],
    )
    k: int = Field(
        default     = 5,
        ge          = 1,
        le          = 10,
        description = "Number of songs to recommend",
    )


class Song(BaseModel):
    track_name:   str
    track_artist: str
    reason:       str


class RecommendResponse(BaseModel):
    mood:            str
    recommendations: list[Song]
    removed_count:   int = Field(description="Songs removed by hallucination guard")
    fallback_notice: str = Field(default="", description="Notice shown when requested genre is not in dataset")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":    "Music Mood Matcher",
        "version": "1.0.0",
        "team":    "The Overfitters",
        "docs":    "/docs",
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "guard_loaded": guard is not None,
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """
    Main recommendation endpoint.

    Pipeline:
      1. Toxicity filter
      2. RAG retrieval + prompt engineering
      3. LM Studio generation
      4. Hallucination guard
    """

    # Step 1 — Toxicity filter
    try:
        toxicity_filter.check(request.mood)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Step 2+3 — RAG + LM Studio
    try:
        recommendations, fallback_notice = generate_recommendations(mood=request.mood, k=request.k)
    except Exception as e:
        log.error(f"Generation error: {e}")
        raise HTTPException(
            status_code=503,
            detail="LM Studio is not reachable. Make sure it is running on localhost:1234.",
        )

    if not recommendations:
        raise HTTPException(
            status_code=500,
            detail="No recommendations generated. Check LM Studio logs.",
        )

    # Step 4 — Hallucination guard (skip pinned songs, they are pre-verified)
    removed_count = 0
    if guard is not None:
        pinned     = [r for r in recommendations if r.get("_pinned")]
        unpinned   = [r for r in recommendations if not r.get("_pinned")]
        unpinned, removed = guard.verify(unpinned)
        removed_count = len(removed)
        recommendations = pinned + unpinned

    if not recommendations:
        raise HTTPException(
            status_code=500,
            detail="All recommendations were hallucinated and removed. Try rephrasing your mood.",
        )

    return RecommendResponse(
        mood            = request.mood,
        recommendations = [Song(**r) for r in recommendations],
        removed_count   = removed_count,
        fallback_notice = fallback_notice,
    )