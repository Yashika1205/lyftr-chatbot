import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import KeyConfig, Settings
from app.main import create_app


def make_settings(tmp_path, **overrides) -> Settings:
    keys = [
        KeyConfig("key_1", "test-secret-one", "gemini/gemini-3.8-flash"),
        KeyConfig("key_2", "test-secret-two", "gemini/gemini-3.8-flash"),
    ]
    base = dict(
        keys=keys,
        models=["gemini/gemini-3.8-flash"],
        database_path=str(tmp_path / "app.db"),
        allowed_origins=["http://localhost:3000"],
        enable_failure_simulation=True,
        default_system_prompt="You are a test assistant.",
        cache_type="local",
        cache_dir=str(tmp_path / "cache"),
        cache_ttl_seconds=3600,
        request_timeout_seconds=5,
        max_history_messages=40,
        mock_llm=True,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(autouse=True)
def _fresh_litellm_worker():
    """Each TestClient runs its own event loop; LiteLLM's global logging worker
    keeps a queue bound to the first loop, so reset it between tests."""
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    GLOBAL_LOGGING_WORKER._queue = None
    GLOBAL_LOGGING_WORKER._worker_task = None
    yield
    GLOBAL_LOGGING_WORKER._queue = None
    GLOBAL_LOGGING_WORKER._worker_task = None


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def new_session():
    return lambda: f"s-{uuid.uuid4().hex[:10]}"


@pytest.fixture
def unique():
    """A message text that no other test has sent (so it can't be a cache hit)."""
    return lambda prefix="msg": f"{prefix} {uuid.uuid4().hex}"
