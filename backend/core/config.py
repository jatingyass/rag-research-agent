"""
backend/core/config.py
Central configuration — reads from .env automatically via pydantic-settings.
"""
from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM (Google Gemini - Free Tier) ────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    
    # ── Embeddings (Free Sentence Transformers) ────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_type: str = "local"

    # ── Re-ranking (Optional) ──────────────────────────
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-english-v3.0"

    # ── Vector DB ──────────────────────────────────────
    vector_db: Literal["chroma", "pinecone"] = "chroma"
    chroma_persist_dir: str = "./chroma_db"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "rag-research-agent"
    pinecone_environment: str = "us-east-1"

    # ── Database ───────────────────────────────────────
    database_url: str = "sqlite:///./rag_agent.db"

    # ── App ────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 7860
    log_level: str = "INFO"

    # ── Security ───────────────────────────────────────
    # When set, all /api/* requests must include X-API-Key: <value>.
    # Leave empty in development to disable authentication.
    api_key: str = ""

    # ── Retrieval ──────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_semantic: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 6

    # ── CORS ───────────────────────────────────────────
    cors_origins: str = "*"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — import this everywhere."""
    return Settings()
