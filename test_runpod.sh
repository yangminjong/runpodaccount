#!/bin/bash

# RunPod API 테스트 스크립트
ENDPOINT_ID="your_endpoint_id_here"
API_KEY="your_api_key_here"

# 테스트 요청
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "question": "회사명: 케이이노텍, 장표구분: 매입매출전표, 거래일자: 2022/01/03, 거래처: sk브로드밴드, 거래구분: 매입, 거래유형: 과세, 금액: 33,680원, 부가세: 3,368원, 합계: 37,048원, 결제방법: 외상, 적요: 1월 데이터요금",
      "use_cot": true,
      "max_new_tokens": 1200,
      "temperature": 0.7,
      "top_p": 0.9
    }
  }'