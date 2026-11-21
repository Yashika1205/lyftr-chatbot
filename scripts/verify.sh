#!/usr/bin/env bash
# Walk through every contract case against a running backend.
# Usage: ./scripts/verify.sh [BASE_URL]   (default http://localhost:8000)
set -u
BASE="${1:-http://localhost:8000}"
BASE="${BASE%/}"
RUN="$(date +%s)"
S1="verify-$RUN-a"; S2="verify-$RUN-b"; S3="verify-$RUN-c"
MSG="What is the capital of France? (verify run $RUN)"

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
post() { # $1 session, $2 message, $3 optional X-Simulate-Failure
  local hdr=()
  [ -n "${3:-}" ] && hdr=(-H "X-Simulate-Failure: $3")
  curl -s -w '\n   -> HTTP %{http_code}\n' -X POST "$BASE/chat" \
    -H 'Content-Type: application/json' "${hdr[@]}" \
    -d "{\"session_id\":\"$1\",\"message\":\"$2\"}"
}

step "GET /health";                              curl -s -w '\n   -> HTTP %{http_code}\n' "$BASE/health"
step "1st message, new session  (expect LIVE_INFERENCE)"; post "$S1" "$MSG"
step "Same message, NEW session (expect LLM_CACHE, same reply, key_id null, lower latency)"; post "$S2" "$MSG"
step "Follow-up in session 1    (expect memory of turn 1, more input_tokens)"; post "$S1" "What city did I just ask about?"
step "Failover key_1=429        (expect 200, key_id key_2, fallback_attempts 1)"; post "$S1" "Name one famous museum there." "key_1=429"
step "Failover key_1=500        (expect 200, key_id key_2)"; post "$S3" "Say hello in French. ($RUN)" "key_1=500"
step "All keys fail *=429       (expect 503 all providers unavailable)"; post "$S1" "This should not be stored" "*=429"
step "Empty message             (expect 422)"; post "$S1" "   "
step "Unknown session history   (expect 404)"; curl -s -w '\n   -> HTTP %{http_code}\n' "$BASE/sessions/does-not-exist-$RUN/messages"
step "History of session 1      (expect 3 user + 3 assistant turns, no failed message)"
curl -s -w '\n   -> HTTP %{http_code}\n' "$BASE/sessions/$S1/messages"
