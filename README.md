# 📹 RDP Video Surveillance AI

## Introduction

This project explores a proof-of-concept architecture for AI-powered video surveillance of remote desktop sessions (RDP), designed to enhance operational oversight and compliance in regulated environments. By leveraging AI models like OmniParser (Florence-2), the system can automatically detect and analyze UI elements, user actions, and potential security threats in RDP session recordings.

## Scenario Overview

The core challenge addressed is the lack of scalable resources to manually review recorded RDP sessions, which are critical for auditing vendor activity and ensuring regulatory compliance. The proposed solution automates the ingestion, enrichment, and analysis of video data captured from remote desktop sessions.

## Architecture Highlights

- **Ingestion Layer**: Periodic or scheduled crawling of RDP session video files; not in scope of this repo
- **Storage & Enrichment**: Videos are sliced, annotated, and enriched with metadata to support downstream analysis
- **Post-Processing**: On-demand querying and summarization via a chat interface or UI, powered by AI search and kernel memory (planned; see [Known limitations / Roadmap](#known-limitations--roadmap))
- **AI Integration**: Object recognition models (e.g., Florence) identify UI elements and user actions within video frames
- **Security Focus**: Supports both detection (e.g., anomaly identification) and prevention (e.g., session termination) strategies; prevention is not in scope of this repo
- **Compliance Alignment**: Designed to align with regulatory requirements, including audit trails and risk classification; not in scope of this repo

## Key Components

### OmniParser (Florence-2)

OmniParser is the core AI model used in this project to detect and understand UI elements in RDP session videos:

- Uses Florence-2 vision models to detect UI icons and elements in desktop screenshots
- Capable of understanding the context and purpose of UI elements
- Allows for tracking user interactions with applications

For more information:
- [OmniParser GitHub Repository](https://github.com/microsoft/OmniParser)
- [Microsoft Research: OmniParser v2](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)

### Orchestrator Service

The orchestrator service manages the overall workflow:
- Video ingestion and processing
- Frame extraction and analysis
- Integration with AI agents for specialized analysis
- Storage and retrieval of results

### AI Agents

The system includes specialized AI agents for different types of analysis:
- **Chat Completion Agent** (`agents/chat_completion`): Analyzes each extracted frame (image + OmniParser output) and produces a structured per-frame result. A natural-language chat/query interface is planned but not yet exposed.
- **Image Analyzer Agent** (`agents/image_analyzer`): Standalone frame analyzer (under development); shares the same prompt format as the Chat Completion agent.
- **Security Reviewer Agent** (`agents/security_reviewer`): Summarizes all per-frame results into one session report with a manual-review flag, `risk_score` (0-100), `risk_level` (low/medium/high/critical) and per-observation evidence (frame, timestamp, confidence).

**Important Configuration for GPT model**

 - gpt-4.1 or gpt-4o with 450K Rate limit (Tokens per minute)
 - Apply a custom Content Filtering based on your content


## Azure Integration

For production deployment, this system leverages various Azure services:

- **Azure Storage**: For storing video files and extracted frames
- **Azure App Services**: For hosting the orchestrator and web interface
- **Azure PostgreSQL**: For storing metadata and analysis results
- **Azure Container Registry**: For managing Docker images
- **N-Series GPU VMs**: For high-performance model inference

## Project Structure

```
rdp-video-surveillance/
├── docker-compose.yml      # omniparser (GPU) + orchestrator services
├── requirements.txt        # Python dependencies for local development
├── .env.example            # Template for .env (all environment variables, with comments)
├── .env.docker.example     # docker compose flavored variant of the template
├── demo.sh                 # bash demo: ingest -> poll job -> print summary -> ask a question
├── demo.cmd                # Windows demo: submits a video to the async ingest endpoint
├── images/
│   └── windows_desktop.png # Sample screenshot used by test/test.py (.mp4 files are git-ignored; bring your own)
├── src/
│   ├── omniparser/         # OmniParser (Florence-2) icon/UI element detection service
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── server.py       # FastAPI server (port 8081)
│   │   ├── setup_weights.cmd / setup_weights.sh  # Download model weights into ./weights
│   │   ├── test_image/
│   │   ├── util/           # OmniParser helpers
│   │   └── weights/        # Model weights (git-ignored, created by setup_weights)
│   └── orchestrator/       # Orchestration service
│       ├── app.py          # FastAPI app (port 8082): endpoints + background pipeline
│       ├── models.py       # Pydantic request/response models
│       ├── errors.py       # Structured error responses
│       ├── retry_utils.py  # Retry/backoff helpers for Azure calls
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── agents/         # AI agents and their prompts (agent.yml)
│       │   ├── chat_completion/
│       │   ├── image_analyzer/
│       │   └── security_reviewer/
│       └── postgresql/     # Database access layer (videos, frames, jobs, pgvector)
└── test/
    ├── test.py             # Smoke test for the OmniParser /parse/ endpoint
    └── demo_flow.py        # End-to-end check: health -> ingest -> poll job -> video -> ask
```

## Requirements

- Docker with Docker Compose v2 (`docker compose`)
- NVIDIA GPU + NVIDIA Container Toolkit (the `omniparser` service reserves one GPU; CPU works only with a modified compose file and is very slow)
- Azure resources (no local fallback in this prototype):
  - Storage account (connection string) for videos and extracted frames
  - Azure OpenAI with a vision-capable chat deployment (e.g. `gpt-4.1` / `gpt-4o`, ~450K TPM) and an embedding deployment (e.g. `text-embedding-ada-002`)
  - Azure Database for PostgreSQL with the `vector` (pgvector) extension allow-listed; tables are created automatically when the orchestrator starts
- A sample RDP recording (`.mp4`). None is shipped with the repo (`*.mp4` is git-ignored).
- Python 3.12+ only if you want to run `test/test.py` or develop locally (`pip install -r requirements.txt`)

## Demo walkthrough

### 1. Clone and download OmniParser weights (one-time, ~GBs)

bash:
```bash
git clone https://github.com/hyssh/rdp-video-surveillance.git && cd rdp-video-surveillance
pip install "huggingface_hub[cli]"
(cd src/omniparser && bash setup_weights.sh)
```

Windows (cmd):
```cmd
git clone https://github.com/hyssh/rdp-video-surveillance.git
cd rdp-video-surveillance
pip install "huggingface_hub[cli]"
cd src\omniparser && setup_weights.cmd && cd ..\..
```

### 2. Configure the environment

bash:
```bash
cp .env.example .env
```
Windows (cmd / PowerShell):
```cmd
copy .env.example .env
```
Edit `.env` and replace every `####` placeholder (storage, Azure OpenAI, PostgreSQL). `docker compose` reads `.env` automatically. `.env.docker.example` is an equivalent variant (use it with `docker compose --env-file .env.docker up`). `.env.example` documents every variable, including the optional tuning knobs.

### 3. Start the services

```bash
docker compose up --build
```
The first start takes several minutes (image build + model load). Orchestrator: `http://localhost:8082`, OmniParser: `http://localhost:8081`.

### 4. Health checks

bash:
```bash
curl http://localhost:8082/            # {"message":"Orchestrator API is running"}
curl http://localhost:8082/healthcheck # orchestrator can reach OmniParser
curl http://localhost:8081/health      # OmniParser is up
```
Windows: use `curl.exe` (Windows 10+) with the same URLs; in PowerShell plain `curl` is an alias of `Invoke-WebRequest`. Interactive API docs: `http://localhost:8082/docs` (FastAPI Swagger UI; OmniParser has its own at `http://localhost:8081/docs`).

### 5. Ingest a video (async)

No sample recording ships with the repo (`*.mp4` is git-ignored): bring a 20-60 second RDP capture. `POST /v2/ingest-video/` takes a multipart form: `file` (required) plus optional `container_name` (default `videos`) and `frames_container` (default `frames`). It returns **202 immediately** with a job id; the analysis runs in the background.

bash:
```bash
curl -X POST "http://localhost:8082/v2/ingest-video/" -F "file=@/path/to/session.mp4"
```
Windows cmd:
```cmd
curl.exe -X POST "http://localhost:8082/v2/ingest-video/" -F "file=@C:\path\to\session.mp4"
```
Windows PowerShell:
```powershell
curl.exe -X POST "http://localhost:8082/v2/ingest-video/" -F "file=@C:\path\to\session.mp4"
```

Response (HTTP 202):
```json
{"job_id": "0b5c...-uuid", "status": "queued", "status_url": "/jobs/0b5c...-uuid"}
```

> Shortcut: `./demo.sh /path/to/session.mp4` (bash, WSL or Git Bash) runs steps 5-8 end to end (override `BASE_URL`, `POLL_SECONDS`, `TIMEOUT_SECONDS`; optional 2nd argument is the question). `demo.cmd C:\path\to\session.mp4` only submits the video and prints the job info. `python test/demo_flow.py /path/to/session.mp4` runs the same flow as a PASS/FAIL check (without a video it only checks health and error shapes, and skips with a message if the orchestrator is unreachable).

### 6. Poll the job

```bash
curl http://localhost:8082/jobs/<job_id>
```
Windows: `curl.exe http://localhost:8082/jobs/<job_id>`

```json
{"job_id": "...", "status": "analyzing", "stage": "analyzing_frames", "frames_done": 7, "frames_total": 20,
 "frames_failed": 0, "video_id": 1, "error": null, "created_at": "...", "updated_at": "..."}
```
`status` moves `queued -> extracting -> analyzing -> summarizing -> done` (or `failed`, with `error` set). Poll every 5-10 seconds until `done`; then use `video_id`.

**Expected wait:** one frame is sampled every 3 seconds, and each frame goes through OmniParser (GPU), a vision LLM call and an embedding call. Plan for roughly 5-20 seconds per frame depending on GPU and Azure latency (not benchmarked), i.e. a 1-minute recording (~20 frames) takes a few minutes. Up to `MAX_CONCURRENT_JOBS` jobs run at once; the rest stay `queued`.

A job can be `done` with `frames_failed > 0`: failed frames are skipped, and the job only fails if every frame fails. Such jobs carry a human-readable `warnings` list (e.g. "N of M frames failed analysis and were skipped"); `status` stays `done`. `demo.sh` and `test/demo_flow.py` print any warnings.

Legacy synchronous mode: `POST /v2/ingest-video/?wait=true` blocks until finished and returns 200 `{"status","video_id","frames":{...},"frames_failed","job_id"}`. Use a long client timeout (`--max-time 3600`).

### 7. List and inspect videos

```bash
curl "http://localhost:8082/videos?limit=10"          # newest first
curl http://localhost:8082/videos/1                   # summary for one video
curl "http://localhost:8082/videos/1/frames?limit=50" # per-frame analysis, ordered by timestamp
```
Windows: same URLs with `curl.exe` (quote URLs containing `&` or `?` in cmd).

`GET /videos/{id}` returns `{id, filename, summary, security_level, frame_count, frames_described, created_at}`. `summary` is the security reviewer's JSON **as a string**: parse it, then start with `manual_review` (Yes/No + reason) and `risk_level`/`risk_score` (0-100: low 0-24, medium 25-49, high 50-74, critical 75-100). Each entry in `analysis_results.key_observations[]` carries `frame_id`, `frame_timestamp`, `evidence`, `confidence` (0-1) and its own risk score, so you can jump to the exact moment in the recording. "not observable" means the model could not confirm the item from the frames. `security_level` is set from the review's `risk_level` (low/medium/high/critical; stays `low` if the summary could not be parsed).

`GET /videos/{id}/frames` items: `{id, frame_timestamp (seconds into the video), image_description, url}`.

### 8. Ask questions about a video

`POST /videos/{id}/ask` embeds the question, retrieves the `top_k` (1-20, default 5) most similar frames and answers from them.

bash:
```bash
curl -X POST http://localhost:8082/videos/1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Were any passwords or keys visible on screen?", "top_k": 5}'
```
Windows cmd (escape the inner quotes):
```cmd
curl.exe -X POST http://localhost:8082/videos/1/ask -H "Content-Type: application/json" -d "{\"question\": \"Were any passwords or keys visible on screen?\", \"top_k\": 5}"
```
Windows PowerShell (use a here-string or a file to avoid quoting problems):
```powershell
'{"question": "Were any passwords or keys visible on screen?", "top_k": 5}' | Set-Content ask.json
curl.exe -X POST http://localhost:8082/videos/1/ask -H "Content-Type: application/json" --data-binary "@ask.json"
```

Response:
```json
{"video_id": 1, "question": "...", "answer": "Yes, at 00:06 ...",
 "matches": [{"frame_id": 2, "frame_timestamp": 6, "image_description": "...", "similarity": 0.87, "url": "https://..."}],
 "note": null}
```
`matches` are the evidence: check the cited frame (`url`) and `frame_timestamp`. If answer generation fails or is disabled, `answer` is `null` and `note` explains why; the matches are still returned.

Sample auditor questions:

1. Does this session require manual review, and why?
2. Were passwords, keys, or other secrets visible on screen? In which frames/timestamps?
3. Did the user install software or run unfamiliar scripts or CLI commands?
4. Did the user log in to or connect to another system or server during the session?
5. Was any data moved off the machine (download, upload, copy to external location), and what was the evidence and confidence?

## API reference

Orchestrator, `http://localhost:8082` (interactive docs at `/docs`).

| Method | Path | Purpose | Success response |
|---|---|---|---|
| GET | `/` | Liveness | 200 `{"message": ...}` |
| GET | `/healthcheck` | Orchestrator can reach OmniParser | 200 |
| POST | `/v2/ingest-video/` | Submit a video (multipart `file`; form `container_name`, `frames_container`; query `wait=false`) | 202 `{job_id, status, status_url}`; with `wait=true` 200 `{status, video_id, frames, frames_failed, job_id}` |
| GET | `/jobs/{job_id}` | Job progress | 200 `{job_id, status, stage, frames_done, frames_total, frames_failed, warnings[], video_id, error, created_at, updated_at}` |
| GET | `/videos?limit&offset` | List videos (newest first; `limit` 1-200) | 200 `{items[{id, filename, security_level, created_at}], total, limit, offset}` |
| GET | `/videos/{id}` | One video with summary | 200 `{id, filename, summary, security_level, frame_count, frames_described, created_at}` |
| GET | `/videos/{id}/frames?limit&offset` | Frames with analysis (`limit` 1-500) | 200 `{items[{id, frame_timestamp, image_description, url}], total, limit, offset}` |
| POST | `/videos/{id}/ask` | Ask about a video (JSON `{question, top_k}`) | 200 `{video_id, question, answer, matches[{frame_id, frame_timestamp, image_description, similarity, url}], note}` |

OmniParser, `http://localhost:8081`: `GET /health`, `POST /parse/` (image in, parsed UI elements out; returns 429 + `Retry-After` when busy).

Errors from the orchestrator use one shape: `{"error": "<code>", "stage": "<where>", "detail": "<message>", "job_id": "<optional>"}` with HTTP 400 (bad input / missing config, e.g. `missing_config`, `invalid_job_id`), 404 (`job_not_found`, `video_not_found`), 502 (upstream Azure/OmniParser error) or 503 (dependency unavailable, e.g. `omniparser_unavailable`). Request-validation failures (e.g. missing `file`, empty `question`, `top_k` out of range) return HTTP 422 with the same shape: `error: "invalid_request"`, `stage: "validating_request"`.

### Smoke tests

```bash
python test/test.py                          # OmniParser only
python test/demo_flow.py [/path/to/video.mp4] # orchestrator flow (set ORCHESTRATOR_URL, JOB_TIMEOUT_SECONDS if needed)
```
`test/test.py` posts `images/windows_desktop.png` to `/parse/` ten times (5 s connect / 120 s read timeout) and prints a PASS/FAIL line; it reads `OMNIPARSER_HOST`/`OMNIPARSER_PORT` from `.env` (default `localhost:8081`).

### Running services individually (optional)

```bash
docker build -t rdp-video-surveillance-omniparser src/omniparser
docker build -t rdp-video-surveillance-orchestrator src/orchestrator
docker run --gpus all -p 8081:8081 -v "$(pwd)/src/omniparser/weights:/app/weights" rdp-video-surveillance-omniparser
docker run -p 8082:8082 --env-file .env -e OMNIPARSER_HOST=host.docker.internal rdp-video-surveillance-orchestrator
```

## Tuning knobs

Set in `.env` (documented in `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `MAX_CONCURRENT_JOBS` | 2 | Ingest jobs the orchestrator runs at once; extra jobs wait in `queued` |
| `MAX_CONCURRENT_OMNIPARSER` | 2 | Frames the orchestrator sends to OmniParser at the same time |
| `OMNIPARSER_MAX_CONCURRENCY` | 1 | Parallel inferences the OmniParser server accepts (bounded by GPU memory); extra requests get HTTP 429 with `Retry-After` |
| `OMNIPARSER_READ_TIMEOUT`, `EMBEDDING_BATCH_SIZE`, `RETRY_MAX_ATTEMPTS` | see `.env.example` | Timeouts, embedding batching and Azure retry behaviour |

Throughput is bounded by a single GPU and by Azure OpenAI rate limits; raising concurrency beyond them only produces 429s/retries.

## Known limitations / Roadmap

- **Jobs run in-process**: they execute inside the orchestrator process and are lost on restart. At startup, jobs still non-terminal are marked `failed` with `interrupted by restart`; this is wrong when running several uvicorn workers (one worker would fail another's live jobs), so keep a single worker.
- **No retry endpoint**: a `failed` job (or a video with `frames_failed > 0`) must be re-ingested by uploading again; blobs are stored as `{job_id}/{filename}`, so re-uploads never overwrite.
- **Stale-job cleanup is partial**: only the restart sweep above; jobs that hang while the process stays up are not detected or removed.
- **No UI**: API (and `/docs`) only.
- **No authentication** on the APIs (demo only).
- **Azure-only**: Blob Storage, Azure OpenAI and PostgreSQL are required; there is no local mode.
- **Frame sampling**: fixed 1 frame / 3 s; short-lived on-screen events can be missed.
- **Model output is not ground truth**: verdicts come from an LLM; always check the cited frame/timestamp before acting.

## License

This project is licensed under the terms found in the [LICENSE](LICENSE) file.


