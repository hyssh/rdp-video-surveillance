FROM python:3.12-slim

WORKDIR /app

# Set noninteractive to avoid timezone prompts
ENV OMNIPARSER_HOST=localhost
ENV OMNIPARSER_PORT=8081

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir semantic-kernel

# Copy application code
COPY . .

# Expose port for the application
EXPOSE 8082

# Command to run the application
CMD ["python", "app.py"]

ARG PORT=8082
ARG OMNIPARSER_HOST=${OMNIPARSER_HOST}
ARG OMNIPARSER_PORT=${OMNIPARSER_PORT}


EXPOSE ${PORT}

CMD ["uvicorn", "server:app", "--host", "localhost", "--port", "8081"]