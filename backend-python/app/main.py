from __future__ import annotations

from time import perf_counter

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import (
    FeedbackRequest,
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    RecommendationItem,
    ReindexRequest,
    ReindexResponse,
)
from app.services.state import AppState

logger = structlog.get_logger(__name__)
app = FastAPI(title="ScholarEvolve API", version="0.2.0")
_state: AppState | None = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    state = get_state()
    return HealthResponse(
        status="ok",
        index_size=len(state.paper_lookup),
        rust_core_loaded=hasattr(state.rust, "module"),
    )


@app.post("/reindex", response_model=ReindexResponse)
async def reindex(req: ReindexRequest) -> ReindexResponse:
    state = get_state()
    started = perf_counter()
    count = await state.reindex(query=req.query, max_results=req.max_results)
    elapsed = perf_counter() - started
    logger.info("reindex_completed", indexed_papers=count, took_seconds=elapsed)
    return ReindexResponse(indexed_papers=count, took_seconds=elapsed)


@app.post("/feedback")
async def feedback(req: FeedbackRequest) -> dict[str, str]:
    state = get_state()
    if req.user_id.strip() == "":
        raise HTTPException(status_code=400, detail="user_id must not be empty")
    if req.paper_id not in state.paper_lookup:
        raise HTTPException(status_code=404, detail=f"paper_id {req.paper_id} not found in index")
    await run_in_threadpool(state.update_feedback, req)
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    state = get_state()
    if not state.retriever.is_ready():
        raise HTTPException(status_code=409, detail="Index not ready. Call /reindex first.")

    reranked, took_ms = await run_in_threadpool(state.recommendation_service.recommend, req)
    response_items: list[RecommendationItem] = []
    for row in reranked:
        paper = state.paper_lookup[row["paper_id"]]
        response_items.append(
            RecommendationItem(
                paper_id=paper.paper_id,
                title=paper.title,
                abstract=paper.abstract,
                authors=paper.authors,
                categories=paper.categories,
                published_date=paper.published_date,
                score=row["score"],
                reasons=row["reasons"],
                metadata=paper.metadata,
            )
        )

    logger.info("recommend_completed", took_ms=took_ms, result_count=len(response_items), user_id=req.user_id)
    return RecommendResponse(
        user_id=req.user_id,
        query=req.user_query,
        took_ms=took_ms,
        recommendations=response_items,
    )
