from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    CLICK = "click"
    BOOKMARK = "bookmark"
    SKIP = "skip"


class RecommendRequest(BaseModel):
    user_query: str
    user_id: str | None = None
    history: list[str] = Field(default_factory=list)
    clicked_papers: list[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)
    ablation_profile: str | None = None


class FeedbackRequest(BaseModel):
    user_id: str
    paper_id: str
    event_type: EventType
    timestamp: datetime


class ReindexRequest(BaseModel):
    max_results: int = Field(default=1000, ge=50, le=10000)
    query: str = "cat:cs.AI OR cat:cs.LG"


class RecommendationItem(BaseModel):
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published_date: datetime
    score: float
    reasons: dict[str, float]
    metadata: dict[str, Any]


class RecommendResponse(BaseModel):
    user_id: str | None
    query: str
    took_ms: float
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    index_size: int
    rust_core_loaded: bool


class ReindexResponse(BaseModel):
    indexed_papers: int
    took_seconds: float
