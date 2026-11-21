# LLM Chatbot with Key Failover and Response Caching

A full-stack chat app: **React** frontend, **FastAPI + LiteLLM** backend, **SQLite** storage.

- Multi-turn chat with history stored per session in SQLite
- A pool of API keys with automatic failover on 429 / quota / 5xx errors
- LiteLLM response cache, labelled `LLM_CACHE` or `LIVE_INFERENCE` on every reply
- Per-request token usage, model, latency, key used and failover count shown next to every reply
- An `X-Simulate-Failure` header to show failover on demand

```
/backend            FastAPI app, LiteLLM gateway, SQLite store, pytest suite
/frontend           React (Vite) single-page app, served by nginx in Docker
/scripts/verify.sh  Walks through every API contract case with curl
docker-compose.yml  Both services + named volume mounted at /data
Makefile            make up / make down (+ logs, test, clean)
.env.example        Every environment variable, with placeholder values
render.yaml         Optional Render Blueprint for deploying both tiers
```

---

## 1. Live URLs

| Tier | URL |
|---|---|
| Frontend | https://lyftr-chatbot-swart.vercel.app |
| Backend | https://lyftr-chatbot-5ww3.onrender.com (`/health`, `/chat`, `/sessions/{id}/messages`) |

The backend runs on Render's free plan, which sleeps after ~15 minutes idle: the first request after a pause can take up to a minute while it wakes.

### How it was deployed

**Backend on Render** (Docker web service)
1. Push this repo to GitHub. In Render: **New → Web Service →** pick the repo, **Root directory** `backend`, **Runtime** Docker.
2. Environment variables:
   - `LLM_API_KEYS` = two (or more) Gemini API keys, comma-separated (free keys from Google AI Studio)
   - `LLM_MODEL` = `gemini/gemini-3.5-flash-lite`
   - `DATABASE_URL` = `sqlite:////data/app.db`
   - `ENABLE_FAILURE_SIMULATION` = `true`
   - `ALLOWED_ORIGINS` = your frontend URL, e.g. `https://your-app.vercel.app` (no trailing slash)
3. Health check path: `/health`. Render sets `PORT`; the container honours it.
   (On Render's free plan the disk is ephemeral; attach a persistent disk at `/data`, or use Railway / Fly.io volumes, to keep history across redeploys.)

**Frontend on Vercel** (or Netlify / Render static)
1. **New Project →** same repo, **Root directory** `frontend`, framework **Vite**.
2. Environment variable `VITE_API_BASE_URL` = the backend URL from above.
3. Deploy, then put the frontend URL into the backend's `ALLOWED_ORIGINS` and redeploy the backend.

Alternatively, `render.yaml` deploys both tiers from one Render Blueprint.

---

## 2. Run locally: with Docker and without

### With Docker

```bash
cp .env.example .env      # then put two real keys in LLM_API_KEYS
make up                   # backend http://localhost:8000, frontend http://localhost:3000
make down                 # stop (the SQLite volume is kept)
```

No keys yet? Set `MOCK_LLM=true` in `.env` to get deterministic offline replies. Caching, failover, tokens and history all still work, so the whole flow can be tried without any API key.

Other targets: `make logs`, `make test`, `make clean` (also deletes the data volume).

### Without Docker

Backend (Python 3.11+):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export LLM_API_KEYS="key-one,key-two" DATABASE_URL="sqlite:///./data/app.db" \
       ALLOWED_ORIGINS="http://localhost:3000" ENABLE_FAILURE_SIMULATION=true
uvicorn app.main:app --reload --port 8000
pytest -q                 # 42 tests, run offline in MOCK_LLM mode
```

Frontend (Node 20+):

```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev    # http://localhost:3000
```

---

## 3. Design decisions: key fallback, caching, SQLite session storage

### Request flow

```
POST /chat
  ├─ validate (Pydantic)                 → 422 on bad input, nothing stored
  ├─ load session: system prompt + history from SQLite
  ├─ build messages = [system, ...history, new user message]
  ├─ LiteLLM cache lookup (model + full message list)
  │     hit  → LLM_CACHE, stored reply + stored usage, key_id = null
  ├─ miss → for key in pool (key_1, key_2, …):
  │          litellm.acompletion(model, messages, api_key=key, num_retries=0)
  │          429 / quota / 5xx / timeout / connection / auth error → next key
  │          success → write to LiteLLM cache, LIVE_INFERENCE
  │     every key failed → 503 {"detail": "all providers unavailable"}, nothing stored
  └─ success → store user + assistant message in one SQLite transaction
```

### Key fallback

- `LLM_API_KEYS` becomes an ordered pool `key_1, key_2, …`. Each key can serve the same model or a different one (`LLM_KEY_MODELS`), so the pool can span models.
- Every provider call goes through `litellm.acompletion`, which normalises errors from any provider into typed exceptions (`RateLimitError`, `ServiceUnavailableError`, `InternalServerError`, `Timeout`, …). The gateway classifies them by status code and message: 429 and quota errors (`RESOURCE_EXHAUSTED`, "quota", "rate limit") and any 5xx, timeout or connection error move on to the next key. A revoked or invalid key (401/403) also fails over, so one bad key never breaks the app.
- **Model fallback inside a key:** if a key's model is rejected for model-specific reasons (retired, "high demand" 503, per-model free-tier quota, timeout), the same key tries the next model in `LLM_FALLBACK_MODELS` before the pool moves to the next key. An auth error (401/403) skips the key's remaining models straight away. The model that last worked is tried first on every key, so a degraded model costs time only once. `fallback_attempts` still counts failed **keys**, as the contract defines it.
- **Diagnostics:** `/health` also returns `models` and `last_provider_error` (time, key id, model, status and the provider's message, with keys scrubbed). The contract allows extra fields, so a failing deployment can be diagnosed without opening the server logs.
- LiteLLM's own retries are off (`num_retries=0`): retrying a rate-limited key wastes time, and moving to the next key is faster.
- **Why an explicit loop instead of LiteLLM's Router fallbacks:** the contract needs the exact `key_id` that answered and the number of keys that failed first (`fallback_attempts`), and it needs failures injected per key for one request only. An explicit loop over `litellm.acompletion` gives exact numbers for both while LiteLLM still does the provider calls, error mapping and caching.
- **Failure simulation:** when `ENABLE_FAILURE_SIMULATION=true`, `X-Simulate-Failure` (`key_1=429`, `key_1=429,key_2=500`, `*=503`) makes the listed keys raise the same LiteLLM exception a real provider failure would, for that request only, so it takes exactly the same failover path. Requests with the header skip the cache read so the key pool is always exercised (otherwise a cached prompt would answer without touching any key). Unsupported statuses or malformed entries return 422. Without the flag the header is ignored.
- Every error leaves the API as JSON with a `detail` field. A catch-all handler turns anything unexpected into `{"detail": "internal server error"}`, never a stack trace. API keys are never logged or returned, only their ids.

### Caching

- LiteLLM's built-in `Cache` is used, **disk-backed** by default (`CACHE_TYPE=disk`, stored under `/data/llm_cache` on the same Docker volume), so cache hits also survive restarts. `CACHE_TYPE=local` switches to in-memory.
- **Cache key = model + the full message list** (system prompt + history + new message), which is LiteLLM's own key for these inputs. So the same first message in a new session is a hit, while the same text later in a conversation, or with a different system prompt, is a miss. A different message can never come from cache.
- The cache is read first through `litellm.cache.async_get_cache` and written right after a successful live call through `litellm.cache.async_add_cache`. LiteLLM's automatic write happens in a background task, so an identical request sent a moment later could miss. Writing explicitly makes the very next identical request a guaranteed hit.
- A hit returns the identical reply text and the **token usage stored with the cached response**, with `key_id: null` and `fallback_attempts: 0`. Hits typically take 1–5 ms versus hundreds of ms for a live call.

### SQLite session storage

- `DATABASE_URL` (e.g. `sqlite:////data/app.db`) points at a file on the named Docker volume `chat_data` mounted at `/data`, so history survives `make down` / `make up` and container restarts.
- The schema is created on startup: the required `messages` table, plus a `sessions` table (session id, the session's system prompt, timestamps) and an index on `(session_id, id)`.
- The client sends only the new message; the backend rebuilds context from SQLite (last `MAX_HISTORY_MESSAGES`, default 40, to bound prompt size).
- The user message and assistant reply are written **in one transaction only after the LLM succeeded**, so a 422 or 503 never leaves half a turn in history. Assistant rows store the full metadata JSON; the system prompt is never part of the history endpoint.
- WAL mode plus one short connection per operation keeps it safe under FastAPI's thread pool with no external services.

### Frontend

- A fresh `session_id` is generated on every page load; "New chat" starts another one.
- Each assistant reply shows `meta-source`, `meta-model`, `meta-latency`, `meta-input-tokens`, `meta-output-tokens` (plus key id and a failover badge).
- 422, 503 and network errors appear as readable red messages in the chat, with a retry link for 503/network errors.
- The backend URL is read at runtime from `config.js` (written by the container entrypoint from `API_BASE_URL`), so one Docker image works with any backend; on Vercel it comes from `VITE_API_BASE_URL` at build time.

---

## 4. Verification: how to see failover, cache hits and per-message token usage

One command runs every case below against any backend:

```bash
./scripts/verify.sh https://<your-backend-url>     # or with no argument for localhost
```

### Cache hit

```bash
B=http://localhost:8000
curl -s -X POST $B/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-a","message":"What is the capital of France?"}'
# "resolution_source":"LIVE_INFERENCE", "key_id":"key_1", latency in the hundreds of ms

curl -s -X POST $B/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-b","message":"What is the capital of France?"}'
# "resolution_source":"LLM_CACHE", same reply, same tokens, "key_id":null, latency a few ms
```

In the UI: send a message, reload the page (new session), send the same message. The badge turns green `LLM_CACHE`.

### Failover

```bash
curl -s -X POST $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: key_1=429' \
  -d '{"session_id":"s-a","message":"Name one museum there."}'
# 200, "key_id":"key_2", "fallback_attempts":1

curl -s -w '\n%{http_code}\n' -X POST $B/chat -H 'Content-Type: application/json' \
  -H 'X-Simulate-Failure: *=429' -d '{"session_id":"s-a","message":"hello"}'
# 503 {"detail":"all providers unavailable"}; the history of s-a is unchanged
```

In the UI: **Settings → Simulate key failure** sends the header with every message. Failover replies get an amber "↻ 1 failover" badge and `key key_2`; `*=429` shows the red "providers unavailable" message.

### Per-message token usage

Every reply carries `input_tokens` (system + history + new message for *this* request) and `output_tokens` (this reply only), never a running total. In one session, `input_tokens` grows turn by turn because history is resent; two equal-length first messages in two new sessions report about the same `input_tokens`. Shown in the UI as `in N · out M tokens` on each reply.

### History, validation, 404

```bash
curl -s $B/sessions/s-a/messages              # oldest first, assistant rows include metadata
curl -s $B/sessions/nope/messages             # 404 {"detail":"session not found"}
curl -s -X POST $B/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-a","message":"   "}'   # 422
```

### Automated tests

`backend/tests/test_api.py`: 42 tests covering the response shape, cache hit/miss rules (new session, same session, different system prompt), identical reply and stored tokens on hits, cache faster than live, per-request tokens, multi-turn memory, session isolation, history survives restart, every 422 case, malformed JSON, 404, failover for 429/500/502/503, 503 with nothing stored, simulation per request only and off when disabled, real LiteLLM `RateLimitError` failover, and no leaked exceptions or secrets.

---

## 5. Setup Used

- **AI tools:** Claude (Anthropic), used to break down the brief, generate and review the code, and write the tests and this README.
- **Stack:** Python 3.12, FastAPI, LiteLLM 1.77.7, SQLite; React 18 + Vite 6; Docker / Docker Compose; nginx for the frontend runtime image.

---

## 6. Bonus features

- **Failure simulation in the UI:** a settings dropdown sends `X-Simulate-Failure` with each message, with a visible banner while it's on, so failover is easy to demo without curl.
- **Disk-backed LiteLLM cache on the data volume:** cache hits survive restarts, not just history.
- **Offline mock mode (`MOCK_LLM=true`):** deterministic replies that echo the earlier turn, so reviewers can run the full stack and the test suite with no API keys or quota.
- **Pool across models:** `LLM_KEY_MODELS` assigns a model per key, and `LLM_FALLBACK_MODELS` lets each key fall back across models, so a retired or overloaded model never takes the app down.
- **Self-diagnosing `/health`:** reports the last provider error (keys redacted) and the model list.
- **Extra metadata in the UI:** key id and a failover badge on each reply, plus a health indicator (keys configured, cache on/off).
- **Session system prompt** (`system-prompt-input`): remembered per session in SQLite and part of the cache key.
- **Retry button** on 503 and network errors; live character counter for the 4000-character limit.
- **Runtime-configurable frontend image:** the backend URL is injected at container start, no rebuild needed.
- **`scripts/verify.sh`:** one command to walk through every contract case against any deployment.
- **Hardening:** non-root backend container, Docker healthcheck (frontend waits for a healthy backend), bounded history window, secrets never logged or returned.
