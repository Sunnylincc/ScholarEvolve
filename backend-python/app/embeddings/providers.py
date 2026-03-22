from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Returns row-major float32 embeddings."""


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=64)
        return np.asarray(vectors, dtype=np.float32)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = model_name

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        vectors = [row.embedding for row in response.data]
        return np.asarray(vectors, dtype=np.float32)


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: dict[str, np.ndarray] = {}
        if path.exists():
            packed = np.load(path, allow_pickle=True).item()
            self._cache = {k: np.asarray(v, dtype=np.float32) for k, v in packed.items()}

    def get(self, key: str) -> np.ndarray | None:
        return self._cache.get(key)

    def put(self, key: str, vector: np.ndarray) -> None:
        self._cache[key] = vector.astype(np.float32)

    def persist(self) -> None:
        np.save(self.path, self._cache, allow_pickle=True)
