from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import yaml

from app.config.settings import settings
from app.embeddings.providers import EmbeddingCache, OpenAIEmbeddingProvider, SentenceTransformerProvider
from app.embeddings.service import EmbeddingService
from app.ingest.arxiv import ArxivSource
from app.models.schemas import FeedbackRequest
from app.retrieval.faiss_index import FaissRetriever
from app.services.paper_store import PaperRecord, PaperStore
from app.services.recommendation import RecommendationService
from app.services.rust_core import load_rust_core


class AppState:
    def __init__(self) -> None:
        self.paper_store = PaperStore(settings.data_dir / settings.papers_file)
        self.paper_lookup = self.paper_store.load()
        self.retriever = FaissRetriever(
            settings.data_dir / settings.faiss_index_file,
            settings.data_dir / settings.id_map_file,
        )
        if (settings.data_dir / settings.faiss_index_file).exists():
            self.retriever.load()

        provider = (
            OpenAIEmbeddingProvider(settings.openai_embedding_model)
            if settings.embedding_provider == "openai"
            else SentenceTransformerProvider(settings.embedding_model_name)
        )
        cache = EmbeddingCache(settings.data_dir / settings.embedding_cache_file)
        self.embedding_service = EmbeddingService(provider=provider, cache=cache, batch_size=settings.embedding_batch_size)

        self._profile_path = settings.data_dir / settings.user_profiles_file
        self.user_profiles: dict[str, dict[str, Any]] = self._load_profiles()
        self.feedback_config = yaml.safe_load(settings.scoring_config_path.read_text(encoding="utf-8")).get("feedback", {})

        self._lock = RLock()
        self.rust = load_rust_core()
        self.recommendation_service = RecommendationService(
            embeddings=self.embedding_service,
            retriever=self.retriever,
            paper_lookup=self.paper_lookup,
            user_profiles=self.user_profiles,
        )
        self.source = ArxivSource()

    def _load_profiles(self) -> dict[str, dict[str, Any]]:
        if not self._profile_path.exists():
            return {}
        payload = json.loads(self._profile_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    def _persist_profiles(self) -> None:
        self._profile_path.write_text(json.dumps(self.user_profiles), encoding="utf-8")

    async def reindex(self, query: str, max_results: int) -> int:
        papers = await self.source.fetch(query=query, max_results=max_results)
        texts = [f"{p.title}. {p.abstract}" for p in papers]
        vectors = self.embedding_service.embed_texts(texts)

        records: list[PaperRecord] = []
        ids: list[str] = []
        for paper, vec in zip(papers, vectors, strict=True):
            ids.append(paper.paper_id)
            records.append(
                PaperRecord(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    abstract=paper.abstract,
                    authors=paper.authors,
                    categories=paper.categories,
                    published_date=paper.published_date,
                    embedding=vec.astype(np.float32).tolist(),
                    metadata=paper.metadata,
                )
            )

        with self._lock:
            self.paper_store.save(records)
            self.paper_lookup = {r.paper_id: r for r in records}
            self.retriever.build_flat_index(vectors, ids)
            self.recommendation_service.paper_lookup = self.paper_lookup
        return len(records)

    def update_feedback(self, feedback: FeedbackRequest) -> dict[str, Any]:
        with self._lock:
            profile = self.user_profiles.get(
                feedback.user_id,
                {
                    "topic_weights": {},
                    "topic_impressions": {},
                    "topic_clicks": {},
                    "event_counter": 0,
                    "last_updated_ts": 0.0,
                },
            )
            payload = {
                "event_type": feedback.event_type.value,
                "paper_id": feedback.paper_id,
                "timestamp": feedback.timestamp.timestamp(),
                "user_state": profile,
                "feedback_config": self.feedback_config,
                "categories": self.paper_lookup.get(feedback.paper_id).categories
                if feedback.paper_id in self.paper_lookup
                else [],
            }
            updated = self.rust.update_feedback(payload)
            self.user_profiles[feedback.user_id] = updated
            self._persist_profiles()
            return updated
