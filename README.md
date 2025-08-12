# RunPod 서버리스 회계 모델 API

Hugging Face의 회계 전문 모델을 RunPod 서버리스 환경에서 실행하는 프로젝트입니다.

## 프로젝트 구조

```
runpodAI/
├── handler.py        # RunPod 서버리스 핸들러
├── requirements.txt  # Python 패키지 의존성
├── Dockerfile       # Docker 이미지 빌드 파일
├── runpod.toml     # RunPod 설정 파일
└── 코랩.py         # 원본 Colab 코드
```

## 배포 방법

### 1. Docker 이미지 빌드

```bash
docker build -t your-dockerhub-username/accounting-model:latest .
docker push your-dockerhub-username/accounting-model:latest
```

### 2. RunPod 설정

1. RunPod 콘솔에서 새 서버리스 엔드포인트 생성
2. Docker 이미지 URL 입력
3. 환경 변수 설정:
   - `HF_TOKEN`: Hugging Face 액세스 토큰

### 3. API 사용 예시

```python
import requests

url = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
headers = {
    "Authorization": "Bearer YOUR_RUNPOD_API_KEY",
    "Content-Type": "application/json"
}

data = {
    "input": {
        "question": "회사명: 케이이노텍, 장표구분: 매입매출전표...",
        "use_cot": True  # Chain-of-Thought 사용 여부
    }
}

response = requests.post(url, json=data, headers=headers)
result = response.json()
print(result["output"])
```

## 주요 기능

- **자동 모델 로딩**: 첫 요청 시 모델을 자동으로 다운로드하고 캐싱
- **GPU 지원**: CUDA 사용 가능 시 자동으로 GPU 활용
- **Chain-of-Thought**: 논리적 추론 과정을 포함한 응답 생성 옵션
- **에러 처리**: 안정적인 에러 처리 및 상태 반환

## 환경 변수

- `HF_TOKEN`: Hugging Face 토큰 (필수)
- `CUDA_VISIBLE_DEVICES`: 사용할 GPU 디바이스 (기본값: 0)
- `HF_HOME`: 모델 캐시 디렉토리 (기본값: /workspace/model_cache)

## 성능 최적화

- 모델 사전 로딩으로 콜드 스타트 시간 단축
- 모델 캐싱으로 반복 다운로드 방지
- GPU 메모리 최적화 설정 포함

## 주의사항

- 최소 24GB VRAM이 필요합니다 (RTX 3090, A40, A100 등)
- 첫 실행 시 모델 다운로드로 인해 시간이 걸릴 수 있습니다
- HF_TOKEN은 보안을 위해 RunPod 시크릿에 저장하세요