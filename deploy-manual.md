# RunPod 수동 배포 가이드

## 1. RunPod 웹 콘솔에서 직접 배포

### 방법 1: RunPod 웹 인터페이스 사용
1. [RunPod Console](https://www.runpod.io/console/serverless) 접속
2. 기존 Endpoint 선택 또는 새로 생성
3. **Container Configuration** 섹션에서:
   - Container Image: `registry.runpod.io/yangminjong/runpodaccount:latest`
   - 또는 GitHub 연동된 경우: `registry.runpod.net/yangminjong-runpodaccount-main-dockerfile:latest`
4. **Environment Variables** 설정:
   ```
   HF_TOKEN=your_huggingface_token
   CUDA_VISIBLE_DEVICES=0
   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```
5. **GPU Configuration**:
   - GPU Type: AMPERE_24 (RTX 3090/A4000) 또는 상위
   - Min Workers: 0
   - Max Workers: 3
6. **Deploy** 클릭

### 방법 2: RunPod CLI 사용
```bash
# RunPod CLI 설치
pip install runpod

# API 키 설정
export RUNPOD_API_KEY="your_api_key_here"

# 새 엔드포인트 생성
runpod serverless create \
  --name "exaone-accounting" \
  --image "registry.runpod.net/yangminjong-runpodaccount-main-dockerfile:latest" \
  --gpu-type "AMPERE_24" \
  --min-workers 0 \
  --max-workers 3 \
  --env HF_TOKEN="your_token"

# 기존 엔드포인트 업데이트
runpod serverless update [ENDPOINT_ID] \
  --image "registry.runpod.net/yangminjong-runpodaccount-main-dockerfile:latest"
```

## 2. GitHub Actions 자동 배포 설정

### 필요한 GitHub Secrets 설정:
1. GitHub 저장소 → Settings → Secrets and variables → Actions
2. 다음 시크릿 추가:
   - `RUNPOD_API_KEY`: RunPod API 키
   - `RUNPOD_ENDPOINT_ID`: Serverless Endpoint ID
   - `RUNPOD_REGISTRY_USERNAME`: RunPod 레지스트리 사용자명
   - `HF_TOKEN`: Hugging Face 토큰

### RunPod API 키 얻기:
1. [RunPod Settings](https://www.runpod.io/console/user/settings) 접속
2. API Keys 섹션
3. "Create API Key" 클릭
4. 키 복사 후 GitHub Secrets에 저장

### Endpoint ID 찾기:
1. RunPod Console → Serverless → Your Endpoint
2. URL에서 ID 확인: `https://www.runpod.io/console/serverless/endpoints/[ENDPOINT_ID]`

## 3. 로컬에서 빌드 후 수동 푸시

```bash
# Docker 이미지 빌드
docker build -t runpod-accounting .

# RunPod 레지스트리에 태그
docker tag runpod-accounting registry.runpod.io/[username]/runpodaccount:latest

# RunPod 레지스트리 로그인
docker login registry.runpod.io -u [username] -p [api_key]

# 이미지 푸시
docker push registry.runpod.io/[username]/runpodaccount:latest

# RunPod에서 엔드포인트 업데이트 (웹 콘솔 또는 API)
```

## 4. 배포 확인

### 엔드포인트 테스트:
```python
import runpod

runpod.api_key = "your_api_key"
endpoint = runpod.Endpoint("your_endpoint_id")

# 요청 보내기
run_request = endpoint.run({
    "input": {
        "question": "회사가 사무용품을 10만원에 구입했습니다.",
        "use_cot": True
    }
})

# 결과 확인
print(run_request.output())
```

### 로그 확인:
1. RunPod Console → Serverless → Your Endpoint → Logs
2. 또는 CLI: `runpod serverless logs [ENDPOINT_ID]`

## 5. 문제 해결

### 컨테이너가 시작되지 않는 경우:
- GPU 메모리 부족: 더 큰 GPU 타입 선택 (A40, A100)
- 환경 변수 확인: HF_TOKEN이 올바르게 설정되었는지 확인
- 이미지 풀 실패: 레지스트리 권한 확인

### "remove container" 오류:
```bash
# RunPod Console에서 엔드포인트 재시작
# 또는 CLI:
runpod serverless restart [ENDPOINT_ID]
```

### 메모리 오류:
- runpod.toml의 메모리 설정 증가
- GPU 타입을 A40 또는 A100으로 변경
- 모델 양자화 고려

## 6. GitHub → RunPod 자동 연동

RunPod은 GitHub 저장소와 직접 연동을 지원합니다:

1. RunPod Console → Serverless → Create Endpoint
2. "Container Image" 섹션에서 "Connect GitHub" 선택
3. GitHub 계정 연동 및 저장소 선택
4. 브랜치 선택 (main)
5. Dockerfile 경로 지정 (/)
6. 자동 빌드 및 배포 활성화

이렇게 설정하면 GitHub에 푸시할 때마다 자동으로 새 이미지가 빌드되고 배포됩니다.