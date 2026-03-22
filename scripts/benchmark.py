from __future__ import annotations

import argparse
import statistics
import time

import httpx


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = max(0, min(len(values) - 1, int(round((len(values) - 1) * q))))
    return sorted(values)[idx]


def benchmark(base_url: str, query: str, top_k: int, n: int) -> dict[str, float]:
    client = httpx.Client(timeout=20)
    e2e_ms: list[float] = []
    model_ms: list[float] = []

    for _ in range(n):
        started = time.perf_counter()
        response = client.post(
            f"{base_url}/recommend",
            json={"user_query": query, "top_k": top_k, "user_id": "bench-user"},
        )
        elapsed = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            raise RuntimeError(f"recommend failed: {response.status_code} {response.text}")

        payload = response.json()
        e2e_ms.append(elapsed)
        model_ms.append(float(payload["took_ms"]))

    return {
        "requests": float(n),
        "e2e_p50_ms": statistics.median(e2e_ms),
        "e2e_p95_ms": percentile(e2e_ms, 0.95),
        "model_p50_ms": statistics.median(model_ms),
        "model_p95_ms": percentile(model_ms, 0.95),
        "e2e_avg_ms": statistics.mean(e2e_ms),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency benchmark for ScholarEvolve API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--query", default="graph neural networks for drug discovery")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    print(benchmark(args.base_url, args.query, args.top_k, args.n))
