from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PaperRecord:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published_date: datetime
    embedding: list[float]
    metadata: dict[str, Any]


class PaperStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, records: list[PaperRecord]) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            for r in records:
                payload = {
                    "paper_id": r.paper_id,
                    "title": r.title,
                    "abstract": r.abstract,
                    "authors": r.authors,
                    "categories": r.categories,
                    "published_date": r.published_date.isoformat(),
                    "embedding": r.embedding,
                    "metadata": r.metadata,
                }
                f.write(json.dumps(payload) + "\n")

    def load(self) -> dict[str, PaperRecord]:
        if not self.path.exists():
            return {}
        results: dict[str, PaperRecord] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            rec = PaperRecord(
                paper_id=payload["paper_id"],
                title=payload["title"],
                abstract=payload["abstract"],
                authors=payload["authors"],
                categories=payload["categories"],
                published_date=datetime.fromisoformat(payload["published_date"]),
                embedding=payload["embedding"],
                metadata=payload["metadata"],
            )
            results[rec.paper_id] = rec
        return results
