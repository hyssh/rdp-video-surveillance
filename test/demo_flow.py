"""
End-to-end check of the orchestrator API: health -> ingest -> poll job -> get video -> ask.

Usage:
    python test/demo_flow.py path/to/session.mp4      # full flow (needs Azure services configured)
    python test/demo_flow.py                          # read-only checks only (no video given)

Env: ORCHESTRATOR_URL (default http://localhost:8082), JOB_TIMEOUT_SECONDS (default 1800),
     POLL_SECONDS (default 5).
Exit codes: 0 = all executed checks passed or the orchestrator is unreachable (skipped), 1 = a check failed.
"""
import os
import sys
import time
from pathlib import Path

import requests

BASE_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8082").rstrip("/")
JOB_TIMEOUT = float(os.getenv("JOB_TIMEOUT_SECONDS", "1800"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "5"))
SHORT = (5, 30)        # (connect, read) for quick GETs
UPLOAD = (5, 600)      # video upload
ASK = (5, 120)         # embedding + LLM answer

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def error_shape_ok(resp, code, error=None):
    try:
        body = resp.json()
    except ValueError:
        return False
    return resp.status_code == code and {"error", "detail"} <= set(body) and (error is None or body["error"] == error)


def main() -> int:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    print(f"Orchestrator: {BASE_URL}")

    try:
        root = requests.get(f"{BASE_URL}/", timeout=SHORT)
    except requests.RequestException as exc:
        print(f"SKIPPED: orchestrator not reachable ({exc.__class__.__name__}). Start it with: docker compose up --build")
        return 0
    check("GET /", root.status_code == 200, f"HTTP {root.status_code}")

    health = requests.get(f"{BASE_URL}/healthcheck", timeout=SHORT)
    check("GET /healthcheck (orchestrator -> OmniParser)", health.status_code == 200, f"HTTP {health.status_code}")

    bad_job = requests.get(f"{BASE_URL}/jobs/not-a-uuid", timeout=SHORT)
    check("GET /jobs/<invalid> -> 400 invalid_job_id", error_shape_ok(bad_job, 400, "invalid_job_id"), f"HTTP {bad_job.status_code}")
    missing = requests.get(f"{BASE_URL}/videos/2147483647", timeout=SHORT)
    check("GET /videos/<unknown> -> 404 video_not_found", error_shape_ok(missing, 404, "video_not_found"), f"HTTP {missing.status_code}")

    listing = requests.get(f"{BASE_URL}/videos", params={"limit": 5}, timeout=SHORT)
    list_ok = listing.status_code == 200 and {"items", "total", "limit", "offset"} <= set(listing.json())
    check("GET /videos", list_ok, f"HTTP {listing.status_code}")

    if video is None:
        print("No video path given: skipping ingest/poll/ask. Pass a .mp4 to run the full flow.")
        return finish()
    if not video.is_file():
        check("video file exists", False, str(video))
        return finish()

    with video.open("rb") as handle:
        accepted = requests.post(f"{BASE_URL}/v2/ingest-video/", files={"file": (video.name, handle)}, timeout=UPLOAD)
    body = accepted.json() if accepted.headers.get("content-type", "").startswith("application/json") else {}
    if not check("POST /v2/ingest-video/ -> 202 with job_id", accepted.status_code == 202 and "job_id" in body,
                 f"HTTP {accepted.status_code} {accepted.text[:200] if accepted.status_code != 202 else body.get('job_id')}"):
        return finish()
    job_id = body["job_id"]

    deadline = time.time() + JOB_TIMEOUT
    job = {}
    last = None
    while time.time() < deadline:
        resp = requests.get(f"{BASE_URL}/jobs/{job_id}", timeout=SHORT)
        if resp.status_code != 200:
            check("GET /jobs/{id}", False, f"HTTP {resp.status_code} {resp.text[:200]}")
            return finish()
        job = resp.json()
        progress = (job["status"], job.get("stage"), job.get("frames_done"), job.get("frames_total"))
        if progress != last:
            print(f"  job {job['status']} / {job.get('stage')} frames {job.get('frames_done')}/{job.get('frames_total')} failed={job.get('frames_failed')}")
            last = progress
        if job["status"] in ("done", "failed"):
            break
        time.sleep(POLL_SECONDS)
    for warning in job.get("warnings") or []:
        print(f"  WARNING: {warning}")
    check("job reached 'done'", job.get("status") == "done", job.get("error") or f"last status {job.get('status')}")
    if job.get("status") != "done" or not job.get("video_id"):
        return finish()
    video_id = job["video_id"]

    detail = requests.get(f"{BASE_URL}/videos/{video_id}", timeout=SHORT)
    detail_json = detail.json() if detail.status_code == 200 else {}
    check("GET /videos/{id} has summary and described frames",
          bool(detail_json.get("summary")) and detail_json.get("frames_described", 0) > 0,
          f"frames {detail_json.get('frames_described')}/{detail_json.get('frame_count')}")

    frames = requests.get(f"{BASE_URL}/videos/{video_id}/frames", params={"limit": 5}, timeout=SHORT)
    frames_json = frames.json() if frames.status_code == 200 else {}
    check("GET /videos/{id}/frames", frames.status_code == 200 and frames_json.get("total", 0) > 0, f"total {frames_json.get('total')}")

    asked = requests.post(f"{BASE_URL}/videos/{video_id}/ask", json={"question": "What applications were used?", "top_k": 3}, timeout=ASK)
    ask_json = asked.json() if asked.status_code == 200 else {}
    check("POST /videos/{id}/ask returns matches", asked.status_code == 200 and len(ask_json.get("matches", [])) > 0,
          f"HTTP {asked.status_code}" + (f" note={ask_json.get('note')}" if ask_json.get("note") else ""))
    if ask_json.get("answer"):
        print(f"  answer: {ask_json['answer'][:300]}")
    return finish()


def finish() -> int:
    passed = sum(results)
    print(f"RESULT: {'PASS' if all(results) else 'FAIL'} ({passed}/{len(results)} checks passed)")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
