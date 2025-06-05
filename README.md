# 📹 RDP Video Surveillance AI

## Introduction

This project explores a proof-of-concept (PoC) architecture for AI-powered video surveillance of remote desktop sessions (RDP), designed to enhance operational oversight and compliance in regulated environments.

## Scenario Overview

The core challenge addressed is the lack of scalable resources to manually review recorded RDP sessions, which are critical for auditing vendor activity and ensuring regulatory compliance. The proposed solution automates the ingestion, enrichment, and analysis of video data captured from remote desktop sessions.

## Architecture Highlights

 - Ingestion Layer: Periodic or scheduled crawling of RDP session video files.
- Storage & Enrichment: Videos are sliced, annotated, and enriched with metadata to support downstream analysis.
- Post-Processing: On-demand querying and summarization via a chat interface or UI, powered by AI search and kernel memory.
- AI Integration: Object recognition models (e.g., Florence) identify UI elements and user actions within video frames.
- Security Focus: Supports both detection (e.g., anomaly identification) and prevention (e.g., session termination) strategies.
- Compliance Alignment: Designed to align with regulatory requirements, including audit trails and risk classification.

## Prototyping Approach

The team used Azure subscriptions to rapidly prototype the system, leveraging services such as:

- Azure Storage
- Azure AI Search
- Azure App Services
- Azure PostgreSQL
- Azure Container Registry
- N-Series GPU VMs for model inference

Skeleton code and architectural templates were shared to accelerate development, with a focus on production readiness and DevOps best practices.

## Run OmniParser (Florence-2) model using Docker

 - OmniParser is going to detect icons in the given image 
 - Details can be found [here](https://github.com/microsoft/OmniParser)

**Build docker**

```cmd
# Basic build command
docker build -t rdp-video-surveillance-omniparser -f src/omniparser/Dockerfile src/omniparser
```

**Run docker**

 - Use GPU for the best performance

```cmd
docker run --gpus all -p 8000:8000 rdp-video-surveillance-omniparser
```

**Test**

```cmd
python test/test.py
```



