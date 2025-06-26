# 📹 RDP Video Surveillance AI

## Introduction

This project explores a proof-of-concept architecture for AI-powered video surveillance of remote desktop sessions (RDP), designed to enhance operational oversight and compliance in regulated environments. By leveraging AI models like OmniParser (Florence-2), the system can automatically detect and analyze UI elements, user actions, and potential security threats in RDP session recordings.

## Scenario Overview

The core challenge addressed is the lack of scalable resources to manually review recorded RDP sessions, which are critical for auditing vendor activity and ensuring regulatory compliance. The proposed solution automates the ingestion, enrichment, and analysis of video data captured from remote desktop sessions.

## Architecture Highlights

- **Ingestion Layer**: Periodic or scheduled crawling of RDP session video files; not socpe of this repo
- **Storage & Enrichment**: Videos are sliced, annotated, and enriched with metadata to support downstream analysis
- **Post-Processing**: On-demand querying and summarization via a chat interface or UI, powered by AI search and kernel memory
- **AI Integration**: Object recognition models (e.g., Florence) identify UI elements and user actions within video frames
- **Security Focus**: Supports both detection (e.g., anomaly identification) and prevention (e.g., session termination) strategies; not socpe of this repo
- **Compliance Alignment**: Designed to align with regulatory requirements, including audit trails and risk classification; not socpe of this repo

## Project Structure

```
rdp-video-surveillance/
├── docker-compose.yml      # Docker Compose configuration for multi-container setup
├── requirements.txt        # Python dependencies for the project
├── v1_video20.cmd          # Script for running video processing (20 frames)
├── v2_video20.cmd          # Script for running v2 video processing (20 frames)
├── v2_video300.cmd         # Script for running v2 video processing (300 frames)
├── images/                 # Test video and image samples
│   ├── 20test.mp4
│   ├── 80test.mp4
│   └── windows_desktop.png
├── src/
│   ├── omniparser/        # Florence-2 model for icon/UI element detection
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── server.py      # FastAPI server for the OmniParser
│   │   ├── setup_weights.cmd
│   │   ├── setup_weights.sh
│   │   ├── util/          # Utilities for the OmniParser
│   │   └── weights/       # Model weights for icon detection and captioning
│   └── orchestrator/      # Main orchestration service
│       ├── app.py         # FastAPI application for orchestration
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── agents/        # AI agents for different analysis tasks
│       │   ├── chat_completion/
│       │   ├── image_analyzer/
│       │   └── security_reviewer/
│       └── postgresql/    # Database access layer
└── test/                  # Test scripts
    └── test.py            # Test script for the system
```

## Requirements

- Docker and Docker Compose
- Python 3.12 or later
- NVIDIA GPU with CUDA support (recommended for optimal performance)
- Docker GPU support configured
- Azure AI Foundation Models access (for production deployment)

## Getting Started

### 1. Clone the repository

```cmd
git clone https://github.com/yourusername/rdp-video-surveillance.git
cd rdp-video-surveillance
```

### 2. Set up the Dev environment

```cmd
pip install -r requirements.txt
```

### 3. Build the Docker images

Build the OmniParser image (AI model for UI element detection):
```cmd
docker build -t rdp-video-surveillance-omniparser -f src/omniparser/Dockerfile src/omniparser
```

Build the Orchestrator image (main application service):
```cmd
docker build -t rdp-video-surveillance-orchestrator -f src/orchestrator/Dockerfile src/orchestrator
```

### 4. Run the services

**Option 1: Using Docker Compose (recommended)**
```cmd
docker-compose up
```

**Option 2: Run services individually**

Run OmniParser (GPU-accelerated):
```cmd
docker run --gpus all -p 8081:8081 rdp-video-surveillance-omniparser uvicorn server:app --host 0.0.0.0 --port 8081
```

Run Orchestrator:
```cmd
docker run -p 8082:8082 rdp-video-surveillance-orchestrator uvicorn app:app --host 0.0.0.0 --port 8082
```

### 5. Testing the System

**Test the API**
```cmd
python test/test.py
```

**Test Video Ingestion**
```cmd
curl -X POST "http://localhost:8082/ingest-video/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@images/80test.mp4"
```

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
- **Chat Completion Agent**: Provides natural language interface for querying video content
- **Image Analyzer Agent**: (Under development) Performs detailed analysis of video frames
- **Security Reviewer Agent**: (Under development) Identifies potential security concerns in user activity

## Azure Integration

For production deployment, this system leverages various Azure services:

- **Azure Storage**: For storing video files and extracted frames
- **Azure App Services**: For hosting the orchestrator and web interface
- **Azure PostgreSQL**: For storing metadata and analysis results
- **Azure Container Registry**: For managing Docker images
- **N-Series GPU VMs**: For high-performance model inference

## License

This project is licensed under the terms found in the [LICENSE](LICENSE) file.


