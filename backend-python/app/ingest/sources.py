from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class RawPaper:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published_date: datetime
    metadata: dict[str, str]


class PaperSource(ABC):
    @abstractmethod
    async def fetch(self, query: str, max_results: int) -> list[RawPaper]:
        """Fetches papers from source API."""
