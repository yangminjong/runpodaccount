# CUDA 12.4 베이스 이미지 사용 (코랩과 동일한 CUDA 버전)
FROM nvidia/cuda:12.4.0-cudnn9-runtime-ubuntu22.04

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12를 기본 python3로 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# pip 업그레이드
RUN python3 -m pip install --upgrade pip

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch CUDA 버전 재설치 (코랩과 동일한 버전)
RUN pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# handler.py 복사
COPY handler.py .

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV CUDA_VISIBLE_DEVICES=0

# Hugging Face 캐시 설정 (RunPod 스토리지 사용)
ENV HF_HOME=/runpod-volume/hf_cache
ENV TRANSFORMERS_CACHE=/runpod-volume/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/hf_cache

# 볼륨 마운트 포인트 생성 (RunPod가 자동으로 마운트)
# /runpod-volume은 RunPod에서 자동으로 마운트되는 영구 스토리지
RUN mkdir -p /runpod-volume

# RunPod serverless 실행
CMD ["python", "-u", "handler.py"]