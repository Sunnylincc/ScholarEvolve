# ScholarEvolve V1.5 (Hybrid Python + Rust + FAISS)

ScholarEvolve is an AI-powered academic discovery engine with a production-oriented hybrid stack:
- **Python/FastAPI** for orchestration, API, ingestion, and experiments.
- **Rust (PyO3/maturin)** for reranking, diversity control, and feedback-state updates.
- **FAISS** for ANN retrieval.

## 1) Proposed repo tree

```text
ScholarEvolve/
├── backend-python/
│   ├── app/
│   │   ├── config/
│   │   ├── embeddings/
│   │   ├── ingest/
│   │   ├── models/
│   │   ├── retrieval/
│   │   ├── services/
│   │   └── main.py
│   ├── data/
│   └── requirements.txt
├── core-rust/
│   ├── src/lib.rs
│   ├── Cargo.toml
│   └── pyproject.toml
├── scripts/
│   ├── benchmark.py
│   ├── evaluate.py
│   └── reindex.py
├── tests/
│   ├── integration/
│   └── unit/
└── docs/architecture.md
```

## 2) Core files

### API
- `POST /recommend` => query embedding + FAISS retrieval + Rust rerank with MMR diversity.
- `POST /feedback` => decayed incremental profile update in Rust.
- `GET /health`
- `POST /reindex` => arXiv fetch + embedding + FAISS rebuild.

### Data model
Stored paper fields:
- `paper_id`, `title`, `abstract`, `authors`, `categories`, `published_date`, `embedding`, `metadata`

### Scoring
Rust computes:

```text
final_score =
  w_semantic * semantic_similarity +
  w_topic * personalized_topic_match +
  w_recency * recency_bonus +
  w_quality * paper_quality_proxy +
  w_novelty * novelty_bonus +
  w_exploration * uncertainty_bonus -
  w_dup * duplication_penalty -
  w_overfit * overspecialization_penalty
```

MMR selection then improves diversity on the top candidates.

## 3) Rust/PyO3 integration

- Exposed methods:
  - `rerank_candidates_json(payload_json: str) -> str`
  - `update_feedback_state_json(payload_json: str) -> str`
- Python makes one batched JSON call per request for reranking and one for feedback updates.
- This keeps Python↔Rust crossing coarse-grained and future-proofs payload schema evolution.

## 4) Benchmarks and evaluation

- `scripts/benchmark.py`: e2e and model-latency quantiles against running API.
- `scripts/evaluate.py`: Precision@K, Recall@K, MRR, NDCG, novelty/diversity, adaptation simulation, ablation entrypoint.

## 5) Architecture diagram

See `docs/architecture.md`.

## 6) Exact local setup commands

### Prerequisites
- Python 3.11+
- Rust stable toolchain

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend-python/requirements.txt
pip install maturin
maturin develop --manifest-path core-rust/Cargo.toml
```

### Build index (offline)
```bash
PYTHONPATH=backend-python python scripts/reindex.py --query "cat:cs.AI OR cat:cs.LG" --max-results 500
```

### Run API
```bash
PYTHONPATH=backend-python uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Recommend
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_query":"scientific graph representation learning","user_id":"u1","top_k":10}'
```

### Feedback
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","paper_id":"1234.5678","event_type":"bookmark","timestamp":"2026-03-22T00:00:00Z"}'
```

### Tests
```bash
PYTHONPATH=backend-python:. pytest -q
cargo test --manifest-path core-rust/Cargo.toml
```

### Benchmark
```bash
python scripts/benchmark.py --base-url http://127.0.0.1:8000 --n 50
```
