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
├── v1_video20.cmd          # Legacy Windows sample: calls the removed /ingest-video/ endpoint (hard-coded paths)
├── v2_video20.cmd          # Windows sample: calls /v2/ingest-video/ (hard-coded paths; edit before use)
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
│       ├── app.py          # FastAPI app (port 8082)
│       ├── retry_utils.py  # Retry/backoff helpers for Azure calls
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── agents/         # AI agents and their prompts (agent.yml)
│       │   ├── chat_completion/
│       │   ├── image_analyzer/
│       │   └── security_reviewer/
│       └── postgresql/     # Database access layer (videos, frames, pgvector)
└── test/
    └── test.py             # Smoke test for the OmniParser /parse/ endpoint
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
curl http://localhost:8082/            # {"message":"Orchstrator API is running"}
curl http://localhost:8082/healthcheck # orchestrator can reach OmniParser
curl http://localhost:8081/health      # OmniParser is up
```
Windows: use `curl.exe` (Windows 10+) with the same URLs; in PowerShell plain `curl` is an alias of `Invoke-WebRequest`. Interactive API docs: `http://localhost:8082/docs` (FastAPI Swagger UI; OmniParser has its own at `http://localhost:8081/docs`).

### 5. Ingest a video

`POST /v2/ingest-video/` takes a multipart form: `file` (required) plus optional `container_name` (default `videos`) and `frames_container` (default `frames`) blob containers.

bash:
```bash
curl -X POST "http://localhost:8082/v2/ingest-video/" \
  -F "file=@/path/to/session.mp4" \
  --max-time 3600
```
Windows cmd:
```cmd
curl.exe -X POST "http://localhost:8082/v2/ingest-video/" -F "file=@C:\path\to\session.mp4" --max-time 3600
```
Windows PowerShell:
```powershell
curl.exe -X POST "http://localhost:8082/v2/ingest-video/" -F "file=@C:\path\to\session.mp4" --max-time 3600
```

Response on success (the request only returns after the whole analysis finished):
```json
{"status": "success", "video_id": 1, "frames": {"status": "success", "message": "Inserted 7 frames", "count": 7}}
```
Errors return `{"detail": "..."}` with HTTP 4xx/5xx (e.g. 400 "No image descriptions found", 500 for Azure/DB failures). Note the `video_id`.

**Expected wait:** ingestion is currently **synchronous**. One frame is sampled every 3 seconds, and each frame goes through OmniParser (GPU), a vision LLM call and an embedding call. Plan for roughly 5-20 seconds per frame depending on GPU and Azure latency (not benchmarked), i.e. a 1-minute recording (~20 frames) takes a few minutes. Keep the first demo to a 20-60 second clip and set a long client timeout (`--max-time`).

### 6. Read the result

There is no read endpoint yet (see roadmap); results are stored in PostgreSQL:

| Table / column | Content |
|---|---|
| `videos.video_description` | Security reviewer JSON for the whole session (summary, `manual_review`, `risk_score`, `risk_level`, `analysis_results.key_observations[]`) |
| `frames.image_description` | Per-frame analysis JSON, ordered by `frames.frame_timestamp` (seconds into the video) |
| `videos.security_level` | Not populated by the pipeline yet; ignore |

bash:
```bash
export PGPASSWORD='<password>' PGSSLMODE=require
psql -h <PGHOST> -U <PGUSER> -d postgres -c "select id, filename, video_description from videos where id = 1;"
psql -h <PGHOST> -U <PGUSER> -d postgres -c "select frame_timestamp, left(image_description, 200) from frames where video_id = 1 order by frame_timestamp;"
```
Windows (cmd): `set PGPASSWORD=<password>` and `set PGSSLMODE=require`, then run the same `psql` commands (single line each).

How to read it: start with `manual_review` (Yes/No + reason) and `risk_level`/`risk_score` (0-100: low 0-24, medium 25-49, high 50-74, critical 75-100). Each entry in `key_observations` carries `frame_id`, `frame_timestamp`, `evidence`, `confidence` (0-1) and its own risk score so you can jump to the exact moment in the recording. "not observable" means the model could not confirm the item from the frames.

### 7. Sample auditor questions

Ask these of the stored results (via SQL today, via the planned chat endpoint later):

1. Which sessions require manual review, and why?
2. Were passwords, keys, or other secrets visible on screen? In which frames/timestamps?
3. Did the user install software or run unfamiliar scripts or CLI commands?
4. Did the user log in to or connect to another system or server during the session?
5. Was any data moved off the machine (download, upload, copy to external location), and what was the evidence and confidence?

### Quick API smoke test (OmniParser only)

```bash
python test/test.py
```
It posts `images/windows_desktop.png` to `/parse/` ten times (5 s connect / 120 s read timeout) and prints a PASS/FAIL line. It reads `OMNIPARSER_HOST`/`OMNIPARSER_PORT` from `.env` (default `localhost:8081`).

### Running services individually (optional)

```bash
docker build -t rdp-video-surveillance-omniparser src/omniparser
docker build -t rdp-video-surveillance-orchestrator src/orchestrator
docker run --gpus all -p 8081:8081 -v "$(pwd)/src/omniparser/weights:/app/weights" rdp-video-surveillance-omniparser
docker run -p 8082:8082 --env-file .env -e OMNIPARSER_HOST=host.docker.internal rdp-video-surveillance-orchestrator
```

## Known limitations / Roadmap

- **Synchronous ingestion**: one HTTP request blocks until the whole video is analyzed. Planned: async jobs (submit, then poll status by `video_id`).
- **No read/chat/UI endpoints**: results must be read from PostgreSQL. Planned: report, search and chat endpoints and a simple UI.
- **Azure-only**: Blob Storage, Azure OpenAI and PostgreSQL are required; there is no local mode. No authentication on the APIs (demo only).
- **Frame sampling**: fixed 1 frame / 3 s; short-lived on-screen events can be missed.
- **Scalability**: a single orchestrator instance and a single GPU. Throughput is bounded by OmniParser and Azure OpenAI rate limits. Tuning knobs (set in `.env`, documented in `.env.example`; they apply once the corresponding code is deployed):
  - `MAX_CONCURRENT_OMNIPARSER` - frames the orchestrator sends to OmniParser at the same time (default 2)
  - `OMNIPARSER_MAX_CONCURRENCY` - parallel inferences the OmniParser server accepts, bounded by GPU memory (default 1); extra requests get HTTP 429 with `Retry-After`
  - Related timeouts/batching: `OMNIPARSER_READ_TIMEOUT`, `EMBEDDING_BATCH_SIZE`, `RETRY_MAX_ATTEMPTS`
- **Model output is not ground truth**: verdicts come from an LLM; always check the cited frame/timestamp before acting. The `v1_video20.cmd` script targets a removed endpoint.


## License

This project is licensed under the terms found in the [LICENSE](LICENSE) file.


