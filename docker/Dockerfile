FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    build-essential \
    unzip \
    tar \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Create models directory
RUN mkdir -p /app/models

# Download video-retalking model from Google Drive
RUN cd /app/models && \
    gdown https://drive.google.com/uc?id=1YuYo_z5bVuyhpnCVAIjHipHVq-aQZfZj -O VideoReTalking_Pretrained.zip && \
    unzip VideoReTalking_Pretrained.zip && \
    rm VideoReTalking_Pretrained.zip && \
    mv VideoReTalking_Pretrained/* . && \
    rm -r VideoReTalking_Pretrained

# Create directory for shared data
RUN mkdir -p /shared_data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["tail", "-f", "/dev/null"]
