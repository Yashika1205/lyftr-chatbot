"""LLM gateway: LiteLLM response cache in front of a failover key pool.

Flow for one request:
  1. Look the exact (model + full message list) up in LiteLLM's cache.
     Hit -> return the stored reply and stored usage, key_id=None.
     (Skipped when X-Simulate-Failure is active, so failover is always exercised.)
  2. Miss -> try each key in pool order with litellm.acompletion.
     A 429, quota error, 5xx, timeout or connection error on one key moves on
     to the next key. X-Simulate-Failure injects those errors per key.
  3. First success is written to the LiteLLM cache and returned.
  4. Every key failed -> AllProvidersUnavailable (the API turns it into 503).
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from dataclasses import dataclass
from typing import Any

import litellm
from litellm.caching.caching import Cache

from .config import KeyConfig, Settings
from .simulation import simulated_status_for

logger = logging.getLogger("chatbot.llm")

LIVE_INFERENCE = "LIVE_INFERENCE"
LLM_CACHE = "LLM_CACHE"

# Keep LiteLLM quiet; it never sees our secrets in logs.
litellm.suppress_debug_info = True
litellm.drop_params = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
# LiteLLM's ModelResponse triggers harmless pydantic serializer warnings on cache writes.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


class AllProvidersUnavailable(Exception):
    """Raised when every key in the pool failed for this request."""


class ProviderError(Exception):
    """A failure from one key (real or simulated) that should trigger failover."""

    def __init__(self, status_code: int | None, reason: str) -> None:
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason


@dataclass
class ChatResult:
    reply: str
    resolution_source: str
    model: str
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract(response: Any) -> tuple[str, int, int]:
    """Pull reply text and token usage out of a ModelResponse or its cached dict."""
    choices = _get(response, "choices") or []
    message = _get(choices[0], "message") if choices else None
    content = (_get(message, "content") or "").strip() if message is not None else ""
    usage = _get(response, "usage") or {}
    input_tokens = int(_get(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(_get(usage, "completion_tokens", 0) or 0)
    return content, input_tokens, output_tokens


def _simulated_exception(status: int, key: KeyConfig) -> Exception:
    """Build the same LiteLLM exception a real provider failure would raise."""
    msg = f"simulated HTTP {status} for {key.key_id}"
    provider = key.model.split("/", 1)[0] if "/" in key.model else "openai"
    if status == 429:
        return litellm.RateLimitError(message=msg, llm_provider=provider, model=key.model)
    if status == 503:
        return litellm.ServiceUnavailableError(message=msg, llm_provider=provider, model=key.model)
    exc = litellm.InternalServerError(message=msg, llm_provider=provider, model=key.model)
    exc.status_code = status  # 500 or 502
    return exc


def _classify(exc: Exception) -> ProviderError:
    """Map any provider exception to a failover-worthy ProviderError.

    429 / quota, 5xx, timeouts and connection errors are the expected cases.
    Auth errors (revoked or invalid key) also fail over, so one bad key in the
    pool never takes the app down.
    """
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if status == 429 or "quota" in text or "rate limit" in text or "resource_exhausted" in text:
        return ProviderError(429, "rate limited / quota exhausted")
    if isinstance(exc, (litellm.Timeout, asyncio.TimeoutError)):
        return ProviderError(504, "timeout")
    if isinstance(exc, litellm.APIConnectionError):
        return ProviderError(status or 502, "connection error")
    if isinstance(status, int) and status >= 500:
        return ProviderError(status, "provider server error")
    return ProviderError(status if isinstance(status, int) else None, type(exc).__name__)


def _safe_detail(exc: Exception, secrets: list[str]) -> str:
    """Short provider error text for logs, with any API key scrubbed out."""
    text = " ".join(str(exc).split())[:300]
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


class LLMGateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache_enabled = False
        self._setup_cache()

    # ---- cache ----------------------------------------------------------

    def _setup_cache(self) -> None:
        ttl = self.settings.cache_ttl_seconds
        if self.settings.cache_type == "disk":
            litellm.cache = Cache(type="disk", disk_cache_dir=self.settings.cache_dir, ttl=ttl)
        else:
            litellm.cache = Cache(type="local", ttl=ttl, default_in_memory_ttl=ttl)
        self.cache_enabled = True

    async def _cache_lookup(self, messages: list[dict[str, str]]) -> tuple[str, Any] | None:
        for model in self.settings.models:
            try:
                cached = await litellm.cache.async_get_cache(model=model, messages=messages)
            except Exception:  # a cache problem must never break chat
                logger.exception("cache lookup failed")
                return None
            if cached:
                return model, cached
        return None

    async def _cache_store(self, model: str, messages: list[dict[str, str]], response: Any) -> None:
        try:
            await litellm.cache.async_add_cache(response, model=model, messages=messages)
        except Exception:
            logger.exception("cache write failed")

    # ---- provider call --------------------------------------------------

    def _mock_response(self, key: KeyConfig, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Deterministic offline reply (MOCK_LLM=true) for tests and keyless demos."""
        last = messages[-1]["content"]
        earlier = [m["content"] for m in messages[:-1] if m["role"] == "user"]
        reply = f'(mock) You said: "{last}".'
        if earlier:
            reply += f' Earlier in this chat you said: "{earlier[-1]}".'
        prompt_tokens = litellm.token_counter(model="gpt-3.5-turbo", messages=messages)
        completion_tokens = litellm.token_counter(model="gpt-3.5-turbo", text=reply)
        return {
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": reply}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
            "model": key.model,
        }

    async def _call_key(self, key: KeyConfig, messages: list[dict[str, str]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": key.model,
            "messages": messages,
            "api_key": key.api_key,
            "timeout": self.settings.request_timeout_seconds,
            "num_retries": 0,   # failover is ours: move to the next key, don't retry this one
            "caching": False,   # cache is read before and written after, explicitly
        }
        if self.settings.mock_llm:
            await asyncio.sleep(0.35)  # stand-in for network latency
            kwargs["mock_response"] = self._mock_response(key, messages)
        return await litellm.acompletion(**kwargs)

    # ---- public ---------------------------------------------------------

    async def complete(self, messages: list[dict[str, str]], simulation: dict[str, int]) -> ChatResult:
        # A request carrying X-Simulate-Failure is a test of the key pool, so it
        # skips the cache read and always goes through the providers. Its
        # successful reply is still written to the cache.
        hit = None if simulation else await self._cache_lookup(messages)
        if hit is not None:
            model, cached = hit
            reply, in_tok, out_tok = _extract(cached)
            if reply:
                return ChatResult(reply, LLM_CACHE, model, in_tok, out_tok, None, 0)

        failures = 0
        for key in self.settings.keys:
            try:
                status = simulated_status_for(simulation, key.key_id)
                if status is not None:
                    raise _simulated_exception(status, key)
                response = await self._call_key(key, messages)
                reply, in_tok, out_tok = _extract(response)
                if not reply:
                    raise ProviderError(None, "empty completion")
            except Exception as exc:  # noqa: BLE001 - every provider failure fails over
                err = exc if isinstance(exc, ProviderError) else _classify(exc)
                failures += 1
                logger.warning("key %s failed (%s %s), trying next key | %s",
                               key.key_id, err.status_code, err.reason,
                               _safe_detail(exc, [k.api_key for k in self.settings.keys]))
                continue

            await self._cache_store(key.model, messages, response)
            return ChatResult(reply, LIVE_INFERENCE, key.model, in_tok, out_tok, key.key_id, failures)

        raise AllProvidersUnavailable()
