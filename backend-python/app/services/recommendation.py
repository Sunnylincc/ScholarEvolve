from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import yaml

from app.config.settings import settings
from app.embeddings.service import EmbeddingService
from app.models.schemas import RecommendRequest
from app.retrieval.faiss_index import FaissRetriever
from app.services.paper_store import PaperRecord
from app.services.rust_core import load_rust_core


class RecommendationService:
    def __init__(
        self,
        embeddings: EmbeddingService,
        retriever: FaissRetriever,
        paper_lookup: dict[str, PaperRecord],
        user_profiles: dict[str, dict[str, Any]],
    ) -> None:
        self.embeddings = embeddings
        self.retriever = retriever
        self.paper_lookup = paper_lookup
        self.user_profiles = user_profiles
        self.rust = load_rust_core()
        self.config = yaml.safe_load(settings.scoring_config_path.read_text(encoding="utf-8"))

    def recommend(self, req: RecommendRequest) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        query_vec = self.embeddings.embed_texts([req.user_query])[0]
        candidate_ids, similarities = self.retriever.search(query_vec, settings.retrieval_top_n)
        if not candidate_ids:
            return [], (time.perf_counter() - started) * 1000.0

        now = datetime.now(timezone.utc)
        user_state = self.user_profiles.get(
            req.user_id or "anon",
            {
                "topic_weights": {},
                "topic_impressions": {},
                "topic_clicks": {},
                "event_counter": 0,
                "last_updated_ts": 0.0,
            },
        )

        candidates: list[dict[str, Any]] = []
        topic_impressions = user_state.get("topic_impressions", {})
        topic_clicks = user_state.get("topic_clicks", {})

        for pid, sim in zip(candidate_ids, similarities, strict=True):
            paper = self.paper_lookup[pid]
            age_days = max((now - paper.published_date).days, 0)
            recency = float(np.exp(-age_days / 365.0))
            quality = min(1.0, float(np.log1p(len(paper.authors) + len(paper.categories)) / 3.0))

            prior_impressions = float(sum(topic_impressions.get(cat, 0.0) for cat in paper.categories))
            prior_clicks = float(sum(topic_clicks.get(cat, 0.0) for cat in paper.categories))
            candidates.append(
                {
                    "paper_id": pid,
                    "semantic_similarity": float(sim),
                    "categories": paper.categories,
                    "recency_bonus": recency,
                    "quality_proxy": quality,
                    "embedding": paper.embedding,
                    "prior_impressions": prior_impressions,
                    "prior_clicks": prior_clicks,
                }
            )

        reranked = self.rust.rerank(
            {
                "weights": self.config["weights"],
                "diversity": self.config["diversity"],
                "candidates": candidates,
                "user_state": user_state,
                "top_k": req.top_k,
            }
        )
        took_ms = (time.perf_counter() - started) * 1000.0
        return reranked, took_ms
