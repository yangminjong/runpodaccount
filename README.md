# EXAONE Accounting Model - RunPod Serverless Deployment

파인튜닝된 EXAONE 회계 모델을 RunPod Serverless로 배포하는 프로젝트입니다.

## 📁 프로젝트 구조

```
.
├── handler.py           # RunPod serverless handler
├── Dockerfile          # Docker 이미지 빌드 설정
├── requirements.txt    # Python 패키지 의존성
├── runpod_config.yaml  # RunPod 배포 설정
├── .dockerignore       # Docker 빌드 제외 파일
└── .github/
    └── workflows/
        └── deploy.yml  # GitHub Actions 자동 배포
```

## 🚀 배포 방법

### 1. 수동 배포

#### Step 1: Docker 이미지 빌드
```bash
docker build -t your-dockerhub-username/exaone-accounting .
```

#### Step 2: Docker Hub에 푸시
```bash
docker login
docker push your-dockerhub-username/exaone-accounting
```

#### Step 3: RunPod에서 Serverless Endpoint 생성
1. [RunPod](https://www.runpod.io/) 대시보드 접속
2. Serverless > Create Endpoint 클릭
3. 다음 설정 입력:
   - Container Image: `your-dockerhub-username/exaone-accounting`
   - GPU Type: NVIDIA A100, RTX 6000, A6000, A40, 또는 RTX 4090 중 사용 가능한 GPU
   - Min Workers: 0
   - Max Workers: 5
   - Environment Variables:
     - `HF_TOKEN`: Hugging Face 토큰
     - `HF_HOME`: `/runpod-volume/hf_cache`

### 2. GitHub Actions 자동 배포

#### 필요한 Secrets 설정 (GitHub Repository Settings)
- `DOCKER_USERNAME`: Docker Hub 사용자명
- `DOCKER_PASSWORD`: Docker Hub 비밀번호
- `RUNPOD_API_KEY`: RunPod API 키
- `HF_TOKEN`: Hugging Face 토큰

develop 브랜치에 푸시하면 자동으로 배포됩니다.

## 📡 API 사용법

### 요청 예시
```python
import requests

url = "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/runsync"
headers = {
    "Authorization": "Bearer {YOUR_RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "input": {
        "question": "회사명: 케이이노텍, 장표구분: 매입매출전표, 거래일자: 2022/01/03, 거래처: sk브로드밴드, 거래구분: 매입, 거래유형: 과세, 금액: 33,680원, 부가세: 3,368원, 합계: 37,048원, 결제방법: 외상, 적요: 1월 데이터요금",
        "use_cot": True,  # Chain-of-Thought 사용
        "max_new_tokens": 1200,
        "temperature": 0.7,
        "top_p": 0.9
    }
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

### 응답 예시
```json
{
    "output": {
        "response": "차변: 통신비(판) 33,680원, 차변: 부가세대급금 3,368원, 대변: 미지급금 37,048원",
        "status": "success"
    },
    "status": "COMPLETED"
}
```

## 🔧 환경 변수

- `HF_TOKEN`: Hugging Face 토큰 (필수)
- `HF_HOME`: Hugging Face 캐시 디렉토리 (기본값: `/runpod-volume/hf_cache`)
- `CUDA_VISIBLE_DEVICES`: GPU 디바이스 설정 (기본값: 0)
- `PYTHONUNBUFFERED`: Python 출력 버퍼링 비활성화 (기본값: 1)

## 📊 리소스 요구사항

- **GPU**: 최소 24GB VRAM (모델 크기: ~18.6GB)
  - 지원 GPU: NVIDIA A100, RTX 6000, A6000, A40, RTX 4090
- **CPU**: 8 cores
- **RAM**: 32GB
- **Storage**: 80GB Network Volume (US-WA-1 region)
  - 영구 스토리지로 모델 캐싱
  - `/runpod-volume` 경로에 자동 마운트

## 💰 비용 최적화

RunPod Serverless는 GPU 사용 시간에 대해서만 과금됩니다:
- Min Workers: 0 (요청이 없을 때 GPU 비활성화)
- Max Workers: 5 (동시 처리 가능)
- Scale Down Delay: 60초 (유휴 시간 후 자동 종료)
- Network Volume: ~$0.20/GB/월 (80GB = ~$16/월)
- Region: US-WA-1 (US West - 낮은 지연시간)

## 📝 참고사항

- 모델은 Hugging Face Hub에서 자동으로 다운로드되며 `/runpod-volume/hf_cache`에 캐시됩니다
- 첫 번째 요청 시 모델 로딩으로 인해 시간이 더 걸릴 수 있습니다 (약 18.6GB 다운로드)
- 이후 요청은 모델이 80GB Network Volume에 캐시되어 더 빠르게 처리됩니다
- US-WA-1 리전의 Network Volume을 사용하여 영구 저장 및 빠른 액세스
- 스토리지 용량 모니터링 기능 내장 (남은 공간 체크)

## 🐛 문제 해결

### 모델 로딩 실패
- HF_TOKEN이 올바르게 설정되었는지 확인
- Hugging Face 저장소 접근 권한 확인

### GPU 메모리 부족
- GPU VRAM이 최소 24GB 이상인지 확인
- 지원 GPU: A100, RTX 6000, A6000, A40, RTX 4090
- 다른 프로세스가 GPU를 사용하고 있지 않은지 확인

### 느린 응답 속도
- Cold start 문제일 수 있음 (첫 요청 시)
- Min Workers를 1로 설정하여 항상 하나의 인스턴스 유지 고려
- Network Volume에 모델이 캐시되어 있는지 확인

### 스토리지 부족
- 80GB Network Volume 중 최소 20GB 여유 공간 필요
- 불필요한 캐시 파일 정리: `rm -rf /runpod-volume/hf_cache/hub/*`