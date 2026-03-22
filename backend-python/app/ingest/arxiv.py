from __future__ import annotations

from datetime import datetime, timezone

import feedparser
import httpx

from app.ingest.sources import PaperSource, RawPaper


ARXIV_URL = "https://export.arxiv.org/api/query"


class ArxivSource(PaperSource):
    async def fetch(self, query: str, max_results: int) -> list[RawPaper]:
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.get(ARXIV_URL, params=params)
            response.raise_for_status()

        parsed = feedparser.parse(response.text)
        papers: list[RawPaper] = []
        for entry in parsed.entries:
            paper_id = entry.id.split("/")[-1]
            papers.append(
                RawPaper(
                    paper_id=paper_id,
                    title=entry.title.strip().replace("\n", " "),
                    abstract=entry.summary.strip().replace("\n", " "),
                    authors=[author.name for author in entry.authors],
                    categories=[tag.term for tag in entry.tags] if "tags" in entry else [],
                    published_date=datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ").replace(
                        tzinfo=timezone.utc
                    ),
                    metadata={"link": entry.link},
                )
            )
        return papers
