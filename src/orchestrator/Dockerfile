# Stage 1: build the React UI (src/orchestrator/ui) into /build/static/app via vite's outDir.
# "Dockerfile ui*" keeps the COPY valid when the ui directory does not exist yet (only the Dockerfile is copied then).
FROM node:20-alpine AS ui-build
WORKDIR /build/ui
COPY Dockerfile ui* /build/ui/
RUN mkdir -p /build/static/app && \
    if [ -f package.json ]; then \
        if [ -f package-lock.json ]; then npm ci; else npm install; fi && npm run build; \
    else \
        echo "No UI sources found, skipping UI build"; \
    fi

FROM python:3.12-slim

WORKDIR /app

# Set noninteractive to avoid timezone prompts
ENV OMNIPARSER_HOST=0.0.0.0
ENV OMNIPARSER_PORT=8081

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Built UI (empty when there is nothing to build; /ui then serves the legacy static/index.html)
COPY --from=ui-build /build/static/app /app/static/app

# Expose port for the application
EXPOSE 8082

# Command to run the application
# CMD ["python", "app.py"]
CMD ["uvicorn", "app:orchestrator", "--host", "0.0.0.0", "--port", "8082"]