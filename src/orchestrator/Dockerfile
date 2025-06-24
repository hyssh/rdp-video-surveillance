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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Expose port for the application
EXPOSE 8082

# Command to run the application
# CMD ["python", "app.py"]
CMD ["uvicorn", "app:orchestrator", "--host", "0.0.0.0", "--port", "8082"]