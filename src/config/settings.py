"""Settings configuration using Pydantic."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Find .env file from project root
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # vLLM LLM Configuration
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "Qwen/Qwen3-VL-4B-Instruct-FP8"
    vllm_port: int = 8000
    vllm_max_len: int = 4096
    vllm_gpu_util: float = 0.70

    # Embedding Configuration
    embedding_base_url: str = "http://localhost:8001/v1"
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_port: int = 8001
    embedding_max_len: int = 512
    embedding_gpu_util: float = 0.10

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/chatbot"

    # ChromaDB
    chroma_persist_dir: str = "./src/database/volumes/chromadb"
    chroma_collection_name: str = "bca_terms"

    # Figure Storage
    figures_dir: str = "./data/figures"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8080


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
