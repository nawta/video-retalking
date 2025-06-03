FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    ffmpeg \
    git \
    wget \
    build-essential \
    cmake \
    python3-dev \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    unzip \
    tar \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3.8 /usr/bin/python

# Copy requirements (build context is project root, so specify subdir)
COPY video-retalking/requirements.txt /app/requirements.txt

# # Install Python dependencies
# RUN pip install --upgrade pip setuptools wheel
# RUN pip install --no-cache-dir torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade pip setuptools wheel
# GPU対応版PyTorchをインストール (CUDA 11.3 対応版)
RUN pip install --no-cache-dir torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code for this component only
COPY video-retalking/ /app/

# Create models directory
RUN mkdir -p /app/checkpoints

# ------------------------------
# (NEW) Copy cached models if they exist, otherwise download
# ------------------------------
# ビルドコンテキスト直下にある models_cache ディレクトリを一時領域へコピー
COPY models_cache/ /tmp/models_cache/

RUN set -eux; \
    # 取得対象ファイル一覧
    FILES="30_net_gen.pth BFM.zip DNet.pt ENet.pth expression.mat face3d_pretrain_epoch_20.pth \
           GFPGANv1.3.pth GPEN-BFR-512.pth LNet.pth ParseNet-latest.pth \
           RetinaFace-R50.pth shape_predictor_68_face_landmarks.dat"; \
    for f in $FILES; do \
        if [ -f "/tmp/models_cache/$f" ]; then \
            echo "[CACHE] copying $f"; \
            cp "/tmp/models_cache/$f" "/app/checkpoints/$f"; \
        else \
            echo "[DOWNLOAD] fetching $f"; \
            wget -q "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/$f" -O "/app/checkpoints/$f"; \
        fi; \
    done && \
    # BFM.zip 解凍 (まだ展開されていない場合)
    if [ ! -d /app/checkpoints/BFM ]; then \
        unzip -q -d /app/checkpoints/BFM /app/checkpoints/BFM.zip; \
    fi

# # Another Option: Download video-retalking model from Google Drive
# RUN cd /app/models && \
#     gdown https://drive.google.com/uc?id=1YuYo_z5bVuyhpnCVAIjHipHVq-aQZfZj -O VideoReTalking_Pretrained.zip && \
#     unzip VideoReTalking_Pretrained.zip && \
#     rm VideoReTalking_Pretrained.zip && \
#     mv VideoReTalking_Pretrained/* . && \
#     rm -r VideoReTalking_Pretrained


# Create directory for shared data
RUN mkdir -p /shared_data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["tail", "-f", "/dev/null"]
