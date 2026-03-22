from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class FaissRetriever:
    def __init__(self, index_path: Path, id_map_path: Path) -> None:
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.index: faiss.Index | None = None
        self.id_map: list[str] = []

    def build_flat_index(self, embeddings: np.ndarray, ids: list[str]) -> None:
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, str(self.index_path))
        self.id_map = ids
        self.id_map_path.write_text(json.dumps(ids), encoding="utf-8")
        self.index = index

    def load(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        self.id_map = json.loads(self.id_map_path.read_text(encoding="utf-8"))

    def is_ready(self) -> bool:
        return self.index is not None and len(self.id_map) > 0

    def search(self, query_embedding: np.ndarray, top_n: int) -> tuple[list[str], list[float]]:
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded")
        query = np.ascontiguousarray(query_embedding.astype(np.float32)).reshape(1, -1)
        scores, indices = self.index.search(query, top_n)
        ids = [self.id_map[i] for i in indices[0] if i >= 0]
        sims = [float(s) for s in scores[0][: len(ids)]]
        return ids, sims
