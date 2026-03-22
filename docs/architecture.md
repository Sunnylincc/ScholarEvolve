# ScholarEvolve Architecture

```mermaid
flowchart LR
    subgraph Offline
      A[arXiv Ingestion] --> B[Embedding Service + Cache]
      B --> C[PaperStore JSONL]
      B --> D[FAISS Index Build]
      C --> E[Feature Snapshot]
    end

    subgraph Online
      U[Client] --> API[FastAPI]
      API --> O[Python Orchestration]
      O --> F[FAISS Retrieval Top-N]
      F --> R[Rust Core: Base Scoring + MMR]
      R --> O
      O --> API
      API --> U
      U --> FB[/feedback]
      FB --> API
      API --> RS[Rust Feedback Update + Decay]
      RS --> P[User Profile Store]
    end
```

## Separation of concerns
- **Offline**: ingestion, embedding generation, caching, and FAISS index rebuild.
- **Online**: query embedding, candidate retrieval, Rust reranking with novelty/exploration, and decayed profile updates.

## Critical performance choices
- One batched call from Python to Rust per recommendation request.
- MMR-style diversity in Rust to avoid Python hot loops over candidate comparisons.
- Threadpool offload for CPU-heavy recommendation and feedback paths to keep FastAPI async-safe.
