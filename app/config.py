from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # 忽略 .env 中未在 Settings 声明的字段（如 Redis/Postgres 等）
    )

    # LLM provider & models
    llm_provider: Literal["openai", "hunyuan"] = Field(
        default="hunyuan", env="LLM_PROVIDER"
    )
    llm_model: str = Field(default="hunyuan-turbo", env="LLM_MODEL")
    embedding_model: str = Field(default="hunyuan-embedding", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1024, env="EMBEDDING_DIMENSION")

    # API Keys
    hunyuan_api_key: str | None = Field(default=None, env="HUNYUAN_API_KEY")
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1", env="OPENAI_API_BASE"
    )

    # Hunyuan endpoint（混元 OpenAI 兼容接口）
    hunyuan_api_base: str = Field(
        default="https://api.hunyuan.cloud.tencent.com/v1", env="HUNYUAN_API_BASE"
    )

    # Vector index (FAISS for current phase)
    faiss_index_path: Path = Field(
        default=PROJECT_ROOT / "data" / "indexes" / "faiss_index",
        env="FAISS_INDEX_PATH",
    )
    top_k: int = Field(default=5, env="TOP_K")

    # Langfuse (optional)
    langfuse_public_key: str | None = Field(default=None, env="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, env="LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = Field(default=None, env="LANGFUSE_HOST")

    # Service ports
    ai_engine_port: int = Field(default=8000, env="AI_ENGINE_PORT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load environment variables (from system and .env) and return a cached Settings instance.
    """
    # Ensure .env is loaded for local development; in production, env vars can come from process env.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings()  # type: ignore[call-arg]

