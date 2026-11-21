"""Contract tests for /health, /chat and /sessions/{id}/messages (MOCK_LLM mode)."""

import litellm
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.simulation import SimulationHeaderError, parse_simulation_header
from tests.conftest import make_settings


# ---- /health ----------------------------------------------------------------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["keys_configured"] == 2
    assert body["cache_enabled"] is True


def test_health_503_without_keys(tmp_path):
    app = create_app(make_settings(tmp_path, keys=[], models=[]))
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 503
    assert "detail" in r.json()


# ---- /chat happy path -------------------------------------------------------

def test_chat_live_response_shape(client, new_session, unique):
    sid = new_session()
    r = client.post("/chat", json={"session_id": sid, "message": unique()})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert body["reply"]
    meta = body["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["model"] == "gemini/gemini-3.8-flash"
    assert isinstance(meta["latency_ms"], int) and meta["latency_ms"] >= 0
    assert isinstance(meta["input_tokens"], int) and meta["input_tokens"] > 0
    assert isinstance(meta["output_tokens"], int) and meta["output_tokens"] > 0
    assert meta["key_id"] == "key_1"
    assert meta["fallback_attempts"] == 0


def test_key_secret_never_returned(client, new_session, unique):
    r = client.post("/chat", json={"session_id": new_session(), "message": unique()})
    assert "test-secret" not in r.text


# ---- caching ------------------------------------------------------------------

def test_same_first_message_in_new_session_is_cache_hit(client, new_session, unique):
    text = unique("What is the capital of France?")
    first = client.post("/chat", json={"session_id": new_session(), "message": text}).json()
    second = client.post("/chat", json={"session_id": new_session(), "message": text}).json()

    assert first["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert second["metadata"]["resolution_source"] == "LLM_CACHE"
    assert second["reply"] == first["reply"]
    assert second["metadata"]["key_id"] is None
    assert second["metadata"]["fallback_attempts"] == 0
    # cached usage is the usage stored with the response
    assert second["metadata"]["input_tokens"] == first["metadata"]["input_tokens"]
    assert second["metadata"]["output_tokens"] == first["metadata"]["output_tokens"]
    assert second["metadata"]["latency_ms"] < first["metadata"]["latency_ms"]


def test_different_message_never_from_cache(client, new_session, unique):
    client.post("/chat", json={"session_id": new_session(), "message": unique("a")})
    r = client.post("/chat", json={"session_id": new_session(), "message": unique("b")}).json()
    assert r["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_same_message_with_different_history_is_not_cached(client, new_session, unique):
    text = unique("hello")
    s1 = new_session()
    client.post("/chat", json={"session_id": s1, "message": text})
    # same text again in the SAME session -> history differs -> must be live
    r = client.post("/chat", json={"session_id": s1, "message": text}).json()
    assert r["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_different_system_prompt_is_not_cached(client, new_session, unique):
    text = unique("hi")
    client.post("/chat", json={"session_id": new_session(), "message": text})
    r = client.post(
        "/chat",
        json={"session_id": new_session(), "message": text, "system_prompt": "Talk like a pirate."},
    ).json()
    assert r["metadata"]["resolution_source"] == "LIVE_INFERENCE"


# ---- tokens -------------------------------------------------------------------

def test_tokens_are_per_request_not_running_total(client, new_session):
    # equal-length, plain-English messages (random hex would tokenize unevenly)
    a = client.post("/chat", json={"session_id": new_session(), "message": "Tell me about cats"}).json()
    b = client.post("/chat", json={"session_id": new_session(), "message": "Tell me about dogs"}).json()
    assert a["metadata"]["resolution_source"] == b["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    # two equal-length first messages in two new sessions -> about the same input tokens
    assert abs(a["metadata"]["input_tokens"] - b["metadata"]["input_tokens"]) <= 2


def test_input_tokens_grow_with_history(client, new_session, unique):
    sid = new_session()
    first = client.post("/chat", json={"session_id": sid, "message": unique()}).json()
    second = client.post("/chat", json={"session_id": sid, "message": unique()}).json()
    assert second["metadata"]["input_tokens"] > first["metadata"]["input_tokens"]
    # output tokens describe only this reply, not a total
    assert second["metadata"]["output_tokens"] < first["metadata"]["output_tokens"] * 3


# ---- multi-turn + history endpoint -------------------------------------------

def test_multi_turn_memory_and_history(client, new_session, unique):
    sid = new_session()
    m1, m2 = unique("My name is Deb"), unique("What is my name?")
    client.post("/chat", json={"session_id": sid, "message": m1})
    r2 = client.post("/chat", json={"session_id": sid, "message": m2}).json()
    # the mock echoes the previous user message, proving history was sent to the LLM
    assert m1 in r2["reply"]

    h = client.get(f"/sessions/{sid}/messages")
    assert h.status_code == 200
    data = h.json()
    assert data["session_id"] == sid
    roles = [m["role"] for m in data["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert data["messages"][0] == {"role": "user", "content": m1}
    meta = data["messages"][3]["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert {"input_tokens", "output_tokens"} <= meta.keys()
    assert all(m["role"] != "system" for m in data["messages"])


def test_sessions_are_isolated(client, new_session, unique):
    s1, s2 = new_session(), new_session()
    secret = unique("only-in-s1")
    client.post("/chat", json={"session_id": s1, "message": secret})
    r = client.post("/chat", json={"session_id": s2, "message": unique()}).json()
    assert secret not in r["reply"]
    assert len(client.get(f"/sessions/{s2}/messages").json()["messages"]) == 2


def test_unknown_session_history_404(client):
    r = client.get("/sessions/does-not-exist/messages")
    assert r.status_code == 404
    assert r.json() == {"detail": "session not found"}


def test_history_survives_restart(tmp_path, new_session, unique):
    sid = new_session()
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as c:
        c.post("/chat", json={"session_id": sid, "message": unique()})
    with TestClient(create_app(settings)) as c:
        assert len(c.get(f"/sessions/{sid}/messages").json()["messages"]) == 2


def test_system_prompt_not_stored_in_history(client, new_session, unique):
    sid = new_session()
    client.post("/chat", json={"session_id": sid, "message": unique(), "system_prompt": "Be brief."})
    msgs = client.get(f"/sessions/{sid}/messages").json()["messages"]
    assert all("Be brief." not in m["content"] for m in msgs)


# ---- validation (422) ---------------------------------------------------------

@pytest.mark.parametrize(
    "payload",
    [
        {"message": "hi"},                                  # missing session_id
        {"session_id": "s-1"},                              # missing message
        {"session_id": "", "message": "hi"},                # empty session_id
        {"session_id": "s-1", "message": ""},               # empty message
        {"session_id": "s-1", "message": "    \n\t "},      # whitespace only
        {"session_id": "s-1", "message": "a" * 4001},       # too long
        {"session_id": "x" * 129, "message": "hi"},         # session_id too long
        {"session_id": 123, "message": "hi"},               # wrong type
    ],
)
def test_validation_errors_422(client, payload):
    r = client.post("/chat", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_message_of_4000_chars_after_trim_is_ok(client, new_session):
    r = client.post("/chat", json={"session_id": new_session(), "message": "  " + "a" * 4000 + "  "})
    assert r.status_code == 200


def test_malformed_json_422(client):
    r = client.post("/chat", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_failed_validation_not_added_to_history(client, new_session):
    sid = new_session()
    client.post("/chat", json={"session_id": sid, "message": ""})
    assert client.get(f"/sessions/{sid}/messages").status_code == 404


# ---- failover via X-Simulate-Failure -------------------------------------------

@pytest.mark.parametrize("status", ["429", "500", "502", "503"])
def test_key_1_failure_falls_over_to_key_2(client, new_session, unique, status):
    r = client.post(
        "/chat",
        json={"session_id": new_session(), "message": unique()},
        headers={"X-Simulate-Failure": f"key_1={status}"},
    )
    assert r.status_code == 200
    meta = r.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert "simulated" not in r.json()["reply"].lower()  # a real answer, not a wrapped error


def test_all_keys_fail_503_and_nothing_stored(client, new_session, unique):
    sid = new_session()
    r = client.post(
        "/chat",
        json={"session_id": sid, "message": unique()},
        headers={"X-Simulate-Failure": "*=429"},
    )
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}
    assert client.get(f"/sessions/{sid}/messages").status_code == 404


def test_explicit_all_keys_fail_503(client, new_session, unique):
    r = client.post(
        "/chat",
        json={"session_id": new_session(), "message": unique()},
        headers={"X-Simulate-Failure": "key_1=429,key_2=500"},
    )
    assert r.status_code == 503


def test_failed_turn_mid_session_not_added(client, new_session, unique):
    sid = new_session()
    client.post("/chat", json={"session_id": sid, "message": unique()})
    client.post("/chat", json={"session_id": sid, "message": unique()},
                headers={"X-Simulate-Failure": "*=503"})
    assert len(client.get(f"/sessions/{sid}/messages").json()["messages"]) == 2


def test_simulation_bypasses_cache_even_for_cached_prompt(client, new_session, unique):
    text = unique("cached prompt")
    client.post("/chat", json={"session_id": new_session(), "message": text})
    hit = client.post("/chat", json={"session_id": new_session(), "message": text}).json()
    assert hit["metadata"]["resolution_source"] == "LLM_CACHE"

    r = client.post("/chat", json={"session_id": new_session(), "message": text},
                    headers={"X-Simulate-Failure": "key_1=429"}).json()
    assert r["metadata"]["key_id"] == "key_2"
    assert r["metadata"]["resolution_source"] == "LIVE_INFERENCE"

    sid = new_session()
    r = client.post("/chat", json={"session_id": sid, "message": text},
                    headers={"X-Simulate-Failure": "*=429"})
    assert r.status_code == 503
    assert client.get(f"/sessions/{sid}/messages").status_code == 404


def test_simulation_is_per_request_only(client, new_session, unique):
    client.post("/chat", json={"session_id": new_session(), "message": unique()},
                headers={"X-Simulate-Failure": "key_1=429"})
    r = client.post("/chat", json={"session_id": new_session(), "message": unique()}).json()
    assert r["metadata"]["key_id"] == "key_1"


def test_simulation_ignored_when_disabled(tmp_path, new_session, unique):
    app = create_app(make_settings(tmp_path, enable_failure_simulation=False))
    with TestClient(app) as c:
        r = c.post("/chat", json={"session_id": new_session(), "message": unique()},
                   headers={"X-Simulate-Failure": "*=429"})
    assert r.status_code == 200
    assert r.json()["metadata"]["key_id"] == "key_1"


def test_invalid_simulation_header_422(client, new_session, unique):
    r = client.post("/chat", json={"session_id": new_session(), "message": unique()},
                    headers={"X-Simulate-Failure": "key_1=418"})
    assert r.status_code == 422


# ---- real provider exceptions also fail over ----------------------------------

def test_real_rate_limit_error_fails_over(client, new_session, unique, monkeypatch):
    calls = []
    original = litellm.acompletion

    async def flaky(**kwargs):
        calls.append(kwargs["api_key"])
        if kwargs["api_key"] == "test-secret-one":
            raise litellm.RateLimitError(message="quota exceeded", llm_provider="gemini",
                                         model=kwargs["model"])
        return await original(**kwargs)

    monkeypatch.setattr(litellm, "acompletion", flaky)
    r = client.post("/chat", json={"session_id": new_session(), "message": unique()})
    assert r.status_code == 200
    assert r.json()["metadata"]["key_id"] == "key_2"
    assert r.json()["metadata"]["fallback_attempts"] == 1
    assert calls[-1] == "test-secret-two" and calls.count("test-secret-two") == 1


def test_auth_error_skips_rest_of_key(client, new_session, unique, monkeypatch):
    calls = []
    original = litellm.acompletion

    async def bad_key(**kwargs):
        calls.append((kwargs["api_key"], kwargs["model"]))
        if kwargs["api_key"] == "test-secret-one":
            raise litellm.AuthenticationError(message="API key not valid", llm_provider="gemini",
                                              model=kwargs["model"])
        return await original(**kwargs)

    monkeypatch.setattr(litellm, "acompletion", bad_key)
    r = client.post("/chat", json={"session_id": new_session(), "message": unique()})
    assert r.json()["metadata"]["key_id"] == "key_2"
    assert [c[0] for c in calls] == ["test-secret-one", "test-secret-two"]
    err = client.get("/health").json()["last_provider_error"]
    assert err["key_id"] == "key_1" and err["status"] == 401
    assert "test-secret" not in str(err)


def test_retired_model_falls_back_to_next_model_same_key(client, new_session, unique, monkeypatch):
    original = litellm.acompletion

    async def retired(**kwargs):
        if kwargs["model"] == "gemini/gemini-3.8-flash":
            raise litellm.NotFoundError(message="model no longer available", llm_provider="gemini",
                                        model=kwargs["model"])
        return await original(**kwargs)

    monkeypatch.setattr(litellm, "acompletion", retired)
    sid = new_session()
    r = client.post("/chat", json={"session_id": sid, "message": unique()}).json()
    meta = r["metadata"]
    assert meta["key_id"] == "key_1"
    assert meta["fallback_attempts"] == 0
    assert meta["model"] == "gemini/gemini-3.5-flash-lite"
    # the working model is remembered for the next request
    r2 = client.post("/chat", json={"session_id": sid, "message": unique()}).json()
    assert r2["metadata"]["model"] == "gemini/gemini-3.5-flash-lite"


def test_unexpected_exceptions_never_leak(client, new_session, unique, monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("secret internal stack detail")

    monkeypatch.setattr(litellm, "acompletion", boom)
    r = client.post("/chat", json={"session_id": new_session(), "message": unique()})
    assert r.status_code == 503
    assert "secret internal" not in r.text


# ---- header parser ------------------------------------------------------------

def test_parse_simulation_header():
    assert parse_simulation_header("key_1=429") == {"key_1": 429}
    assert parse_simulation_header(" key_1=429 , key_2=500 ") == {"key_1": 429, "key_2": 500}
    assert parse_simulation_header("*=503") == {"*": 503}
    assert parse_simulation_header("") == {}
    for bad in ["key_1", "key_1=abc", "key_1=404", "=429"]:
        with pytest.raises(SimulationHeaderError):
            parse_simulation_header(bad)
