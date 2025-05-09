FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    build-essential \
    cmake \
    libsndfile1 \
    python3-dev \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install numpy first
RUN pip install --no-cache-dir numpy==1.23.4 gdown

# Copy requirements
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Download model files
RUN mkdir -p /app/checkpoints && \
    cd /app/checkpoints && \
    gdown https://drive.google.com/uc?id=1YuYo_z5bVuyhpnCVAIjHipHVq-aQZfZj -O models.zip && \
    unzip models.zip && \
    rm models.zip

# Create directory for shared data
RUN mkdir -p /shared_data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command - use tail -f /dev/null to keep container running
CMD ["tail", "-f", "/dev/null"]
