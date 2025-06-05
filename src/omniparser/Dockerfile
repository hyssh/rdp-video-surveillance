FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set noninteractive to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV SOM_MODEL_PATH=weights/icon_detect/model.pt
ENV CAPTION_MODEL_NAME=florence2
ENV CAPTION_MODEL_PATH=weights/icon_caption_florence
ENV DEVICE=cuda
ENV BOX_THRESHOLD=0.05
ENV HOST=0.0.0.0
ENV PORT=8000

# Install dependencies and add deadsnakes PPA for Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    build-essential \
    libgl1-mesa-glx \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && gpg --no-tty --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776 \
    && gpg --no-tty --export BA6932366A755776 > /etc/apt/keyrings/deadsnakes.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes-ppa.list \
    && apt-get update

# Install Python 3.12 and pip
RUN apt-get install -y python3.12 python3.12-venv python3.12-dev \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3.12 get-pip.py \
    && rm get-pip.py

# Set python3.12 as default python and pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3 1

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN chmod +x setup_weights.sh && ./setup_weights.sh

ARG PORT=8000

EXPOSE ${PORT}

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]