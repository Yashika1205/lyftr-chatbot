"""FastAPI app exposing /health, /chat and /sessions/{session_id}/messages."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings, load_settings
from .db import Database
from .llm import AllProvidersUnavailable, LLMGateway
from .simulation import SimulationHeaderError, parse_simulation_header

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("chatbot")


# ---- request / response models ---------------------------------------------

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated id, 1-128 chars")
    message: str = Field(..., description="1-4000 chars after trimming")
    system_prompt: Optional[str] = Field(default=None, max_length=8000)

    @field_validator("session_id")
    @classmethod
    def _check_session_id(cls, v: str) -> str:
        v = v.strip()
        if not 1 <= len(v) <= 128:
            raise ValueError("session_id must be 1-128 characters")
        return v

    @field_validator("message")
    @classmethod
    def _check_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        if len(v) > 4000:
            raise ValueError("message must be at most 4000 characters after trimming")
        return v

    @field_validator("system_prompt")
    @classmethod
    def _clean_system_prompt(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None


class ChatMetadata(BaseModel):
    resolution_source: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata


# ---- app -------------------------------------------------------------------

def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    db = Database(settings.database_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.init()
        app.state.gateway = LLMGateway(settings)
        logger.info(
            "started: %d key(s), models=%s, cache=%s, failure_simulation=%s, mock=%s",
            settings.keys_configured, settings.models, settings.cache_type,
            settings.enable_failure_simulation, settings.mock_llm,
        )
        yield

    app = FastAPI(title="LLM Chatbot with Key Failover and Response Caching", lifespan=lifespan)
    app.state.settings = settings
    app.state.db = db

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins or ["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Simulate-Failure"],
    )

    # -- error handlers: always JSON with a "detail" field, never a stack trace --

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        errors = []
        for err in exc.errors():
            err = {k: v for k, v in err.items() if k in ("type", "loc", "msg")}
            errors.append(err)
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logger.exception("unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    # -- routes ------------------------------------------------------------

    @app.get("/health")
    async def health():
        db_ok = await run_in_threadpool(db.ping)
        gateway = getattr(app.state, "gateway", None)
        keys = settings.keys_configured
        if not db_ok or keys == 0 or gateway is None:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "not ready",
                    "status": "error",
                    "database": db_ok,
                    "keys_configured": keys,
                    "cache_enabled": bool(gateway and gateway.cache_enabled),
                },
            )
        return {
            "status": "ok",
            "keys_configured": keys,
            "cache_enabled": gateway.cache_enabled,
            # extra diagnostics (allowed by the contract); never contains a key
            "models": settings.models,
            "last_provider_error": gateway.last_error,
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        body: ChatRequest,
        x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
    ):
        started = time.perf_counter()

        simulation: dict[str, int] = {}
        if settings.enable_failure_simulation and x_simulate_failure:
            try:
                simulation = parse_simulation_header(x_simulate_failure)
            except SimulationHeaderError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        session = await run_in_threadpool(db.get_session, body.session_id)
        # A system_prompt in the request overrides (and is remembered for) this session.
        system_prompt = (
            body.system_prompt
            or (session or {}).get("system_prompt")
            or settings.default_system_prompt
        )
        stored_prompt = body.system_prompt or (session or {}).get("system_prompt")

        history = await run_in_threadpool(
            db.get_history, body.session_id, settings.max_history_messages
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": body.message})

        try:
            result = await app.state.gateway.complete(messages, simulation)
        except AllProvidersUnavailable:
            # Nothing is written to history for a failed request.
            raise HTTPException(status_code=503, detail="all providers unavailable")

        latency_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
        metadata = {
            "resolution_source": result.resolution_source,
            "model": result.model,
            "latency_ms": latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "key_id": result.key_id,
            "fallback_attempts": result.fallback_attempts,
        }
        await run_in_threadpool(
            db.save_exchange, body.session_id, stored_prompt, body.message, result.reply, metadata
        )
        return {"session_id": body.session_id, "reply": result.reply, "metadata": metadata}

    @app.get("/sessions/{session_id}/messages")
    async def session_messages(session_id: str):
        exists = await run_in_threadpool(db.session_exists, session_id)
        if not exists:
            raise HTTPException(status_code=404, detail="session not found")
        messages = await run_in_threadpool(db.get_history, session_id)
        return {"session_id": session_id, "messages": messages}

    return app


app = create_app()
