
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):

    # ==========================
    # Index Configuration
    # ==========================

    INDEX_DIR: Path = (
        PROJECT_ROOT
        / "data/processed/docling/index"
    )

    # ==========================
    # Embedding Configuration
    # ==========================

    EMBEDDING_MODEL: str = (
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 16

    # ==========================
    # Retrieval Configuration
    # ==========================

    RRF_K: int = 60
    CANDIDATE_K: int = 20

    # ==========================
    # Reranker Configuration
    # ==========================

    RERANKER_MODEL: str = (
        "cross-encoder/ms-marco-MiniLM-L6-v2"
    )

    RERANKER_DEVICE: str = "cpu"
    RERANKER_BATCH_SIZE: int = 16

    # ==========================
    # LLM Configuration
    # ==========================

    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = "qwen3.8-flash"

    LLM_TEMPERATURE: float = 0.2
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_RETRIES: int = 1

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
