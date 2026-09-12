from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str | None = None
    weaviate_collection: str = "DocumentChunk"

    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_provider: str = "sentence-transformers"
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_device: str = "cpu"
    embedding_normalize: bool = True
    chunk_size_tokens: int = Field(default=512, gt=0)
    chunk_overlap_tokens: int = Field(default=100, ge=0)
    max_file_size_mb: int = Field(default=5, gt=0)
    allowed_file_types: list[str] = ["pdf", "txt", "md", "csv", "json"]

    metadata_file_path: str = "app/data/metadata.json"
    raw_documents_path: str = "app/data/cache"
    cors_origins: list[str] = ["http://localhost:3000"]

    llm_provider: str = "auto"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-20250514"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"
    gemini_url: str = "https://generativelanguage.googleapis.com/v1beta"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    llm_max_tokens: int = Field(default=1024, gt=0)
    llm_temperature: float = Field(default=0.7, ge=0, le=1)
    llm_timeout_seconds: float = Field(default=30, gt=0)
    llm_retries: int = Field(default=2, ge=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("allowed_file_types", "cors_origins", mode="before")
    @classmethod
    def parse_csv_values(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
