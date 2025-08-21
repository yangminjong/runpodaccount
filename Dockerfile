# NVIDIA 공식 CUDA 이미지 사용 (가장 안정적인 버전)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV LANG=C.UTF-8

# 타임존 설정
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Python 3.10 및 필수 패키지 설치 (Ubuntu 22.04 기본 버전)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10을 기본 python으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

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