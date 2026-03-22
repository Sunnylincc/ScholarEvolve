from __future__ import annotations

import argparse
import asyncio

from app.services.state import AppState


async def main(query: str, max_results: int) -> None:
    state = AppState()
    count = await state.reindex(query=query, max_results=max_results)
    print(f"Indexed {count} papers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="cat:cs.AI OR cat:cs.LG")
    parser.add_argument("--max-results", type=int, default=500)
    args = parser.parse_args()
    asyncio.run(main(args.query, args.max_results))
