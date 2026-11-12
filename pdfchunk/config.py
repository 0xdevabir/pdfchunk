"""Runtime configuration, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_embedding_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    )
    gemini_concurrency: int = field(default_factory=lambda: _int("GEMINI_CONCURRENCY", 4))
    gemini_max_retries: int = field(default_factory=lambda: _int("GEMINI_MAX_RETRIES", 5))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("PDFCHUNK_DATA_DIR", "data")))
    max_upload_mb: int = field(default_factory=lambda: _int("PDFCHUNK_MAX_UPLOAD_MB", 100))

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)


settings = Settings()
