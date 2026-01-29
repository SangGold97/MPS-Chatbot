"""Settings configuration using Pydantic."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # vLLM Configuration
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "Qwen/Qwen3-VL-4B-Instruct" #"Qwen/Qwen3-VL-4B-Thinking-FP8"

    # Embedding Configuration
    embedding_base_url: str = "http://localhost:8001/v1"
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/chatbot"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "documents"

    # Figure Storage
    figures_dir: str = "./data/figures"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
