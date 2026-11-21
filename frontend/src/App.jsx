import { useEffect, useRef, useState } from "react";
import { API_BASE_URL, fetchHealth, newSessionId, sendChat } from "./api.js";

const MAX_CHARS = 4000;

const SIMULATION_PRESETS = [
  { value: "", label: "Off" },
  { value: "key_1=429", label: "key_1 → 429 (rate limit)" },
  { value: "key_1=500", label: "key_1 → 500 (server error)" },
  { value: "key_1=429,key_2=500", label: "key_1 → 429, key_2 → 500" },
  { value: "*=429", label: "* → 429 (every key fails)" },
];

function Metadata({ meta }) {
  const cached = meta.resolution_source === "LLM_CACHE";
  return (
    <div className="meta">
      <span className={`chip source ${cached ? "cache" : "live"}`} data-testid="meta-source">
        {meta.resolution_source}
      </span>
      <span className="chip" title="Model">
        <span className="chip-label">model</span>
        <span data-testid="meta-model">{meta.model}</span>
      </span>
      <span className="chip" title="Server-measured latency">
        <span className="chip-label">latency</span>
        <span data-testid="meta-latency">{meta.latency_ms} ms</span>
      </span>
      <span className="chip" title="Tokens for this request only">
        <span className="chip-label">in</span>
        <span data-testid="meta-input-tokens">{meta.input_tokens}</span>
        <span className="chip-sep">·</span>
        <span className="chip-label">out</span>
        <span data-testid="meta-output-tokens">{meta.output_tokens}</span>
        <span className="chip-label">tokens</span>
      </span>
      <span className="chip subtle" title="API key that answered (null on cache hits)">
        <span className="chip-label">key</span>
        <span data-testid="meta-key-id">{meta.key_id ?? "—"}</span>
      </span>
      {meta.fallback_attempts > 0 && (
        <span className="chip warn" data-testid="meta-fallback" title="Keys that failed before this one answered">
          ↻ {meta.fallback_attempts} failover{meta.fallback_attempts > 1 ? "s" : ""}
        </span>
      )}
    </div>
  );
}

function Message({ item, onRetry }) {
  if (item.role === "user") {
    return (
      <div className="row user">
        <div className="bubble user-bubble" data-testid="user-message">{item.content}</div>
      </div>
    );
  }
  if (item.role === "error") {
    return (
      <div className="row assistant">
        <div className="bubble error-bubble" role="alert" data-testid="error-message">
          <div className="error-title">Couldn't get a reply</div>
          <div>{item.content}</div>
          {item.retry && (
            <button className="link-btn" onClick={() => onRetry(item.retry)}>
              Retry this message
            </button>
          )}
        </div>
      </div>
    );
  }
  return (
    <div className="row assistant">
      <div className="bubble assistant-bubble" data-testid="assistant-message">
        <div className="reply-text">{item.content}</div>
        <Metadata meta={item.metadata} />
      </div>
    </div>
  );
}

export default function App() {
  // A new session starts on every page load.
  const [sessionId, setSessionId] = useState(() => newSessionId());
  const [items, setItems] = useState([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [simulate, setSimulate] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [health, setHealth] = useState(null);
  const listRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    fetchHealth().then(setHealth);
  }, []);

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [items, pending]);

  async function send(text) {
    const message = text.trim();
    if (!message || pending) return;
    setInput("");
    setItems((prev) => [...prev, { role: "user", content: message }]);
    setPending(true);
    try {
      const res = await sendChat({ sessionId, message, systemPrompt, simulateFailure: simulate });
      setItems((prev) => [...prev, { role: "assistant", content: res.reply, metadata: res.metadata }]);
    } catch (err) {
      setItems((prev) => [
        ...prev,
        { role: "error", content: err.message, retry: err.status === 422 ? null : message },
      ]);
    } finally {
      setPending(false);
      inputRef.current?.focus();
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send(input);
    }
  }

  function newChat() {
    setSessionId(newSessionId());
    setItems([]);
    setInput("");
    inputRef.current?.focus();
  }

  const over = input.trim().length > MAX_CHARS;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="logo">✦</div>
          <div>
            <h1>LLM Chatbot</h1>
            <p className="subtitle">Key failover · response caching · per-request tokens</p>
          </div>
        </div>
        <div className="top-actions">
          <span
            className={`status ${health ? (health.ok ? "ok" : "down") : ""}`}
            title={API_BASE_URL}
            data-testid="health-status"
          >
            <span className="dot" />
            {health == null
              ? "Checking…"
              : health.ok
                ? `Online · ${health.keys_configured} keys · cache ${health.cache_enabled ? "on" : "off"}`
                : "Backend unreachable"}
          </span>
          <button className="ghost-btn" onClick={() => setShowSettings((s) => !s)} aria-expanded={showSettings}>
            {showSettings ? "Hide settings" : "Settings"}
          </button>
          <button className="ghost-btn" onClick={newChat} data-testid="new-chat-button">
            New chat
          </button>
        </div>
      </header>

      {showSettings && (
        <section className="settings">
          <label className="field">
            <span>System prompt (this session)</span>
            <textarea
              data-testid="system-prompt-input"
              rows={2}
              placeholder="e.g. You are a friendly travel guide. Keep answers under 3 sentences."
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
            />
          </label>
          <label className="field narrow">
            <span>Simulate key failure (X-Simulate-Failure)</span>
            <select
              data-testid="simulate-failure-select"
              value={SIMULATION_PRESETS.some((p) => p.value === simulate) ? simulate : "__custom"}
              onChange={(e) => setSimulate(e.target.value === "__custom" ? simulate : e.target.value)}
            >
              {SIMULATION_PRESETS.map((p) => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
              {!SIMULATION_PRESETS.some((p) => p.value === simulate) && (
                <option value="__custom">Custom: {simulate}</option>
              )}
            </select>
            <input
              className="custom-sim"
              placeholder="or type e.g. key_2=503"
              value={simulate}
              onChange={(e) => setSimulate(e.target.value)}
            />
          </label>
        </section>
      )}

      {simulate && (
        <div className="sim-banner">
          Failure simulation on for every message: <code>X-Simulate-Failure: {simulate}</code>
          <button className="link-btn" onClick={() => setSimulate("")}>turn off</button>
        </div>
      )}

      <main className="messages" ref={listRef} aria-live="polite">
        {items.length === 0 && !pending && (
          <div className="empty">
            <h2>Start a conversation</h2>
            <p>
              Every reply shows where it came from (<b>LIVE_INFERENCE</b> or <b>LLM_CACHE</b>),
              the model, latency and that request's tokens.
            </p>
            <div className="suggestions">
              {["What is the capital of France?", "Explain caching in one sentence.", "Give me 3 tips for writing cold emails."].map((s) => (
                <button key={s} className="suggestion" onClick={() => send(s)}>{s}</button>
              ))}
            </div>
          </div>
        )}
        {items.map((item, i) => (
          <Message key={i} item={item} onRetry={send} />
        ))}
        {pending && (
          <div className="row assistant">
            <div className="bubble assistant-bubble typing" aria-label="Assistant is typing">
              <span /><span /><span />
            </div>
          </div>
        )}
      </main>

      <footer className="composer">
        <div className={`composer-box ${over ? "over" : ""}`}>
          <textarea
            ref={inputRef}
            data-testid="chat-input"
            rows={1}
            placeholder="Message the assistant…  (Enter to send, Shift+Enter for a new line)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            autoFocus
          />
          <button
            data-testid="send-button"
            className="send-btn"
            onClick={() => send(input)}
            disabled={pending || !input.trim()}
          >
            {pending ? "Sending…" : "Send"}
          </button>
        </div>
        <div className="composer-foot">
          <span>Session <code data-testid="session-id">{sessionId}</code></span>
          <span className={over ? "count over" : "count"}>{input.trim().length} / {MAX_CHARS}</span>
        </div>
      </footer>
    </div>
  );
}
