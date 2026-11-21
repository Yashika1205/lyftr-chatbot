"""Application settings, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def sqlite_path_from_url(url: str) -> str:
    """Turn a SQLAlchemy-style SQLite URL into a filesystem path.

    sqlite:////data/app.db -> /data/app.db   (absolute)
    sqlite:///./app.db     -> ./app.db       (relative)
    A bare path is returned unchanged.
    """
    prefix = "sqlite:///"
    if url.startswith(prefix):
        return url[len(prefix):] or "app.db"
    if url.startswith("sqlite://"):
        raise ValueError(f"Unsupported DATABASE_URL: {url!r}")
    return url


@dataclass(frozen=True)
class KeyConfig:
    key_id: str      # public id: key_1, key_2, ...
    api_key: str     # secret, never returned or logged
    model: str       # LiteLLM model this key serves


@dataclass(frozen=True)
class Settings:
    keys: list[KeyConfig]
    database_path: str
    allowed_origins: list[str]
    enable_failure_simulation: bool
    default_system_prompt: str
    cache_type: str
    cache_dir: str
    cache_ttl_seconds: int
    request_timeout_seconds: float
    max_history_messages: int
    mock_llm: bool
    models: list[str] = field(default_factory=list)          # every model that may answer
    fallback_models: list[str] = field(default_factory=list)  # tried per key after its own model

    @property
    def keys_configured(self) -> int:
        return len(self.keys)


def load_settings() -> Settings:
    api_keys = _csv(os.getenv("LLM_API_KEYS"))
    default_model = os.getenv("LLM_MODEL", "gemini/gemini-3.5-flash-lite").strip()
    # Optional: one model per key (same order as LLM_API_KEYS) to spread the
    # pool across models, e.g. "gemini/gemini-3.8-flash,gemini/gemini-3.5-flash-lite".
    per_key_models = _csv(os.getenv("LLM_KEY_MODELS"))

    keys = [
        KeyConfig(
            key_id=f"key_{i + 1}",
            api_key=key,
            model=per_key_models[i] if i < len(per_key_models) else default_model,
        )
        for i, key in enumerate(api_keys)
    ]

    # Backup models tried on the same key when its primary model is rejected
    # (retired, not on the free tier, per-model quota). Set LLM_FALLBACK_MODELS=""
    # to disable.
    fallback_env = os.getenv("LLM_FALLBACK_MODELS")
    fallback_models = _csv(fallback_env) if fallback_env is not None else [
        "gemini/gemini-3.5-flash-lite",
        "gemini/gemini-3.1-flash-lite",
        "gemini/gemini-3.6-flash",
        "gemini/gemini-3.8-flash",
    ]

    models: list[str] = []
    for name in [k.model for k in keys] + fallback_models:
        if name not in models:
            models.append(name)

    database_url = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")
    db_path = sqlite_path_from_url(database_url)
    default_cache_dir = str(Path(db_path).resolve().parent / "llm_cache")

    return Settings(
        keys=keys,
        models=models,
        fallback_models=fallback_models,
        database_path=db_path,
        allowed_origins=_csv(os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")),
        enable_failure_simulation=_bool(os.getenv("ENABLE_FAILURE_SIMULATION"), False),
        default_system_prompt=os.getenv(
            "DEFAULT_SYSTEM_PROMPT",
            "You are a helpful, concise assistant. Answer clearly and accurately.",
        ),
        cache_type=os.getenv("CACHE_TYPE", "disk").strip().lower(),
        cache_dir=os.getenv("CACHE_DIR", default_cache_dir),
        cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", str(7 * 24 * 3600))),
        request_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "15")),
        max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "40")),
        mock_llm=_bool(os.getenv("MOCK_LLM"), False),
    )
