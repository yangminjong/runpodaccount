#!/bin/bash

# RunPod 배포 스크립트
# 사용법: ./runpod-deploy.sh [endpoint_id] [api_key]

ENDPOINT_ID=${1:-$RUNPOD_ENDPOINT_ID}
API_KEY=${2:-$RUNPOD_API_KEY}

if [ -z "$ENDPOINT_ID" ] || [ -z "$API_KEY" ]; then
    echo "❌ Error: ENDPOINT_ID and API_KEY are required"
    echo "Usage: ./runpod-deploy.sh [endpoint_id] [api_key]"
    echo "Or set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY environment variables"
    exit 1
fi

echo "🚀 Deploying to RunPod Serverless..."
echo "Endpoint ID: $ENDPOINT_ID"

# 새 버전으로 엔드포인트 업데이트
response=$(curl -s -X POST "https://api.runpod.io/v2/serverless/endpoints/${ENDPOINT_ID}/update" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "dockerImage": "registry.runpod.net/yangminjong-runpodaccount-main-dockerfile:latest",
    "gpuType": "AMPERE_24",
    "minWorkers": 0,
    "maxWorkers": 3,
    "idleTimeout": 60,
    "env": {
      "CUDA_VISIBLE_DEVICES": "0",
      "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
    },
    "volumeInGb": 50,
    "volumeMountPath": "/workspace/model_cache"
  }')

echo "Response: $response"

# 배포 상태 확인
echo "⏳ Checking deployment status..."
sleep 5

status=$(curl -s -X GET "https://api.runpod.io/v2/serverless/endpoints/${ENDPOINT_ID}" \
  -H "Authorization: Bearer ${API_KEY}" | jq -r '.status')

echo "📊 Deployment status: $status"

if [ "$status" = "running" ]; then
    echo "✅ Deployment successful!"
else
    echo "⚠️ Deployment may still be in progress. Check RunPod console for details."
fi

# 엔드포인트 URL 표시
echo ""
echo "🔗 Endpoint URL: https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync"
echo ""
echo "📝 Test with:"
echo 'curl -X POST "https://api.runpod.ai/v2/'${ENDPOINT_ID}'/runsync" \'
echo '  -H "Authorization: Bearer '${API_KEY}'" \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"input": {"question": "테스트 질문", "use_cot": true}}'"'"