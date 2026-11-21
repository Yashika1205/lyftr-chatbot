const runtimeBase =
  typeof window !== "undefined" && window.__APP_CONFIG__ && window.__APP_CONFIG__.API_BASE_URL;

export const API_BASE_URL = (
  runtimeBase ||
  import.meta.env.VITE_API_BASE_URL ||
  "http://localhost:8000"
).replace(/\/+$/, "");

export function newSessionId() {
  const rand =
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID().replace(/-/g, "").slice(0, 12)
      : Math.random().toString(36).slice(2, 14);
  return `s-${rand}`;
}

/** Turn any FastAPI error body into one readable sentence. */
function describeError(status, body) {
  const detail = body && body.detail;
  if (status === 503) {
    return "All AI providers are unavailable right now (every API key failed). Please try again in a moment.";
  }
  if (status === 422) {
    if (Array.isArray(detail)) {
      const parts = detail.map((d) => {
        const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : "request";
        const msg = String(d.msg || "is invalid").replace(/^Value error, /, "");
        return `${field}: ${msg}`;
      });
      return `Your message couldn't be sent — ${parts.join("; ")}.`;
    }
    return `Your message couldn't be sent — ${detail || "invalid request"}.`;
  }
  if (typeof detail === "string") return `Error ${status}: ${detail}`;
  return `Something went wrong (HTTP ${status}). Please try again.`;
}

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export async function sendChat({ sessionId, message, systemPrompt, simulateFailure }) {
  const payload = { session_id: sessionId, message };
  if (systemPrompt && systemPrompt.trim()) payload.system_prompt = systemPrompt.trim();

  const headers = { "Content-Type": "application/json" };
  if (simulateFailure && simulateFailure.trim()) {
    headers["X-Simulate-Failure"] = simulateFailure.trim();
  }

  let res;
  try {
    res = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
  } catch {
    throw new ApiError(0, "Can't reach the server. Check your connection and try again.");
  }

  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON body: handled below */
  }

  if (!res.ok) throw new ApiError(res.status, describeError(res.status, body));
  if (!body || typeof body.reply !== "string" || !body.metadata) {
    throw new ApiError(res.status, "The server sent an unexpected response. Please try again.");
  }
  return body;
}

export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    const body = await res.json();
    return { ok: res.ok && body.status === "ok", ...body };
  } catch {
    return { ok: false };
  }
}
