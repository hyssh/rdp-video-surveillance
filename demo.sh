#!/usr/bin/env bash
# End-to-end demo: ingest a recording -> poll the job -> print the security summary -> ask a question.
# Usage: ./demo.sh path/to/session.mp4 ["question to ask"]
# Env:   BASE_URL (default http://localhost:8082), POLL_SECONDS (default 5), TIMEOUT_SECONDS (default 1800)
# Needs: curl, python3 (no jq).
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8082}"
POLL_SECONDS="${POLL_SECONDS:-5}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
VIDEO="${1:-}"
QUESTION="${2:-Were any passwords, keys or other secrets visible on screen? Cite the frames.}"

if [ -z "$VIDEO" ] || [ ! -f "$VIDEO" ]; then
  echo "Usage: $0 path/to/session.mp4 [\"question\"]  (no sample video ships with the repo)" >&2
  exit 2
fi

# json_get '<json>' 'expr on d' -> prints the value (empty if missing)
json_get() {
  python3 -c 'import json,sys
d = json.loads(sys.argv[1])
v = eval(sys.argv[2])
print("" if v is None else v)' "$1" "$2"
}

echo "== Health: $BASE_URL"
if ! curl -sS --max-time 10 -f "$BASE_URL/healthcheck" >/dev/null; then
  echo "Orchestrator or OmniParser is not healthy. Run: docker compose up --build" >&2
  exit 1
fi

echo "== Ingest: $VIDEO"
RESP=$(curl -sS --max-time 600 -X POST "$BASE_URL/v2/ingest-video/" -F "file=@$VIDEO" -w '\n%{http_code}')
CODE=$(printf '%s\n' "$RESP" | tail -n 1)
BODY=$(printf '%s\n' "$RESP" | sed '$d')
if [ "$CODE" != "202" ]; then
  echo "Ingest rejected (HTTP $CODE): $BODY" >&2
  exit 1
fi
JOB_ID=$(json_get "$BODY" 'd["job_id"]')
echo "job_id=$JOB_ID"

echo "== Polling /jobs/$JOB_ID every ${POLL_SECONDS}s (timeout ${TIMEOUT_SECONDS}s)"
START=$(date +%s)
while :; do
  JOB=$(curl -sS --max-time 30 -f "$BASE_URL/jobs/$JOB_ID")
  STATUS=$(json_get "$JOB" 'd["status"]')
  echo "  status=$STATUS stage=$(json_get "$JOB" 'd["stage"]') frames=$(json_get "$JOB" 'd["frames_done"]')/$(json_get "$JOB" 'd["frames_total"]') failed=$(json_get "$JOB" 'd["frames_failed"]')"
  case "$STATUS" in
    done)
      python3 -c 'import json,sys
for w in json.loads(sys.argv[1]).get("warnings") or []:
    print("  WARNING:", w)' "$JOB"
      break ;;
    failed) echo "Job failed: $(json_get "$JOB" 'd["error"]')" >&2; exit 1 ;;
  esac
  if [ $(( $(date +%s) - START )) -gt "$TIMEOUT_SECONDS" ]; then
    echo "Timed out waiting for the job (it may still be running: GET $BASE_URL/jobs/$JOB_ID)" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

VIDEO_ID=$(json_get "$JOB" 'd["video_id"]')
echo "== Video $VIDEO_ID summary"
DETAIL=$(curl -sS --max-time 30 -f "$BASE_URL/videos/$VIDEO_ID")
python3 -c '
import json, sys
d = json.loads(sys.argv[1])
print("file:", d.get("filename"), "| frames described:", d.get("frames_described"), "/", d.get("frame_count"))
raw = d.get("summary") or ""
try:
    s = json.loads(raw)
except ValueError:
    print(raw or "(no summary)")
else:
    print("manual_review:", s.get("manual_review"), "-", s.get("manual_review_reason"))
    print("risk:", s.get("risk_level"), s.get("risk_score"))
    print("summary:", s.get("summary"))
' "$DETAIL"

echo "== Ask: $QUESTION"
PAYLOAD=$(python3 -c 'import json,sys; print(json.dumps({"question": sys.argv[1], "top_k": 5}))' "$QUESTION")
ANSWER=$(curl -sS --max-time 120 -f -X POST "$BASE_URL/videos/$VIDEO_ID/ask" -H "Content-Type: application/json" -d "$PAYLOAD")
python3 -c '
import json, sys
d = json.loads(sys.argv[1])
print("answer:", d.get("answer") or "(none)")
if d.get("note"):
    print("note:", d["note"])
for m in d.get("matches", []):
    print("  frame %s @ %ss similarity=%.3f" % (m["frame_id"], m.get("frame_timestamp"), m["similarity"]))
' "$ANSWER"
