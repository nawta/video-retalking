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

# Model Download
# # Option1: Download video-retalking model from Google Drive
# RUN cd /app/models && \
#     gdown https://drive.google.com/uc?id=1YuYo_z5bVuyhpnCVAIjHipHVq-aQZfZj -O VideoReTalking_Pretrained.zip && \
#     unzip VideoReTalking_Pretrained.zip && \
#     rm VideoReTalking_Pretrained.zip && \
#     mv VideoReTalking_Pretrained/* . && \
#     rm -r VideoReTalking_Pretrained
# TODO: need to link the models to the appropriate location by ln -s .... ...

# # Option2: Download from GitHub Release
RUN mkdir ./checkpoints  
RUN wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/30_net_gen.pth -O ./checkpoints/30_net_gen.pth
RUN wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/BFM.zip -O ./checkpoints/BFM.zip
RUN wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/DNet.pt -O ./checkpoints/DNet.pt
RUN wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ENet.pth -O ./checkpoints/ENet.pth
RUN wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/expression.mat -O ./checkpoints/expression.mat
RUN wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/face3d_pretrain_epoch_20.pth -O ./checkpoints/face3d_pretrain_epoch_20.pth
RUN wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GFPGANv1.3.pth -O ./checkpoints/GFPGANv1.3.pth
RUN wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GPEN-BFR-512.pth -O ./checkpoints/GPEN-BFR-512.pth
RUN wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/LNet.pth -O ./checkpoints/LNet.pth
RUN wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ParseNet-latest.pth -O ./checkpoints/ParseNet-latest.pth
RUN wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/RetinaFace-R50.pth -O ./checkpoints/RetinaFace-R50.pth
RUN wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/shape_predictor_68_face_landmarks.dat -O ./checkpoints/shape_predictor_68_face_landmarks.dat
RUN unzip -d ./checkpoints/BFM ./checkpoints/BFM.zip



# Create directory for shared data
RUN mkdir -p /shared_data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["tail", "-f", "/dev/null"]
