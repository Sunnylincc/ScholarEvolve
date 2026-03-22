from __future__ import annotations

import hashlib

import numpy as np

from app.embeddings.providers import EmbeddingCache, EmbeddingProvider


class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider, cache: EmbeddingCache, batch_size: int = 64) -> None:
        self.provider = provider
        self.cache = cache
        self.batch_size = batch_size

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        output: list[np.ndarray] = []
        missing: list[tuple[int, str, str]] = []
        for idx, text in enumerate(texts):
            key = self._key(text)
            cached = self.cache.get(key)
            if cached is None:
                missing.append((idx, key, text))
                output.append(np.empty(0, dtype=np.float32))
            else:
                output.append(cached)

        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start : start + self.batch_size]
            vectors = self.provider.embed_texts([item[2] for item in chunk])
            for (idx, key, _), vector in zip(chunk, vectors, strict=True):
                self.cache.put(key, vector)
                output[idx] = vector

        self.cache.persist()
        return np.vstack(output).astype(np.float32)
