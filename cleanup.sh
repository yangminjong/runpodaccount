#!/bin/bash

# RunPod 컨테이너 정리 스크립트

echo "🧹 Cleaning up RunPod containers..."

# 실행 중인 컨테이너 정지
echo "Stopping running containers..."
docker stop $(docker ps -q --filter "ancestor=registry.runpod.net/yangminjong-runpodaccount-main-dockerfile*") 2>/dev/null || true

# 모든 관련 컨테이너 삭제
echo "Removing containers..."
docker rm -f $(docker ps -aq --filter "ancestor=registry.runpod.net/yangminjong-runpodaccount-main-dockerfile*") 2>/dev/null || true

# 중복된 컨테이너 이름 정리
docker rm -f runpod-accounting-service 2>/dev/null || true

# 미사용 볼륨 정리
echo "Cleaning up unused volumes..."
docker volume prune -f

# Docker 시스템 정리
echo "Cleaning Docker system..."
docker system prune -f

# GPU 메모리 정리 (NVIDIA GPU가 있는 경우)
if command -v nvidia-smi &> /dev/null; then
    echo "Resetting GPU..."
    nvidia-smi --gpu-reset
fi

echo "✅ Cleanup completed!"