from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RustCoreAdapter:
    module: Any

    def rerank(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        raw = self.module.rerank_candidates_json(json.dumps(payload))
        return json.loads(raw)

    def update_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw = self.module.update_feedback_state_json(json.dumps(payload))
        return json.loads(raw)


class PurePythonFallback:
    def rerank(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = payload["candidates"]
        sorted_candidates = sorted(candidates, key=lambda row: row.get("semantic_similarity", 0.0), reverse=True)
        return [
            {
                "paper_id": row["paper_id"],
                "score": float(row.get("semantic_similarity", 0.0)),
                "reasons": {"semantic": float(row.get("semantic_similarity", 0.0)), "final": float(row.get("semantic_similarity", 0.0))},
            }
            for row in sorted_candidates[: payload.get("top_k", 10)]
        ]

    def update_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = payload.get("user_state", {"topic_weights": {}, "topic_clicks": {}, "topic_impressions": {}, "event_counter": 0, "last_updated_ts": 0.0})
        return state


def load_rust_core() -> RustCoreAdapter | PurePythonFallback:
    try:
        import scholarevolve_core

        return RustCoreAdapter(module=scholarevolve_core)
    except Exception:
        return PurePythonFallback()
