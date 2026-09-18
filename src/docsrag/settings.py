"""Configuracao do servico."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuracao tipada, carregada do ambiente."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",
    )

    corpus_dir: Path = Field(default=Path("corpus"), description="Pasta dos normativos.")
    environment: Literal["dev", "staging", "prod"] = Field(default="dev")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    chunk_max_chars: int = Field(default=1200, gt=0)
    chunk_overlap_chars: int = Field(default=150, ge=0)
    top_k: int = Field(default=5, ge=1, le=50)
    require_citation: bool = Field(default=True)

    min_recall_at_k: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Limiar de recall abaixo do qual a CI falha.",
    )
    min_mrr: float = Field(default=0.6, ge=0, le=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devolve a configuracao carregada uma unica vez por processo."""
    return Settings()
