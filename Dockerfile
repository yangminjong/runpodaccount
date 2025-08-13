# CUDA 11.8과 Python 3.10을 포함한 기본 이미지 사용
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /workspace

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 업그레이드
RUN pip install --upgrade pip

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY handler.py .

# 환경 변수 설정 (HF_TOKEN은 RunPod 시크릿에서 설정)
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 모델 캐시 디렉토리 생성
RUN mkdir -p /workspace/model_cache
ENV HF_HOME=/workspace/model_cache

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import torch; print('Health check passed')" || exit 1

# RunPod 핸들러 실행
CMD ["python", "-u", "handler.py"]