from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCHOLAR_EVOLVE_", env_file=".env")

    app_name: str = "ScholarEvolve"
    data_dir: Path = Path("backend-python/data")
    papers_file: str = "papers.jsonl"
    embedding_cache_file: str = "embedding_cache.npy"
    faiss_index_file: str = "papers.faiss"
    id_map_file: str = "id_map.json"
    user_profiles_file: str = "user_profiles.json"

    embedding_provider: str = "sentence_transformers"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 64

    retrieval_top_n: int = 120
    rerank_top_k: int = 10

    scoring_config_path: Path = Path("backend-python/app/config/scoring.yaml")


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
