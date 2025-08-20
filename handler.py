# -*- coding: utf-8 -*-
import sys
import os
import shutil

# EXAONE4 실제 클래스 로드 (pickle 호환성을 위해)
try:
    import exaone_loader
    print("EXAONE4 classes loaded successfully")
except Exception as e:
    print(f"Warning: Could not load EXAONE4 classes: {e}")

import runpod
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit

# 환경 변수에서 HF 토큰 가져오기 (필수)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required but not set")

# 볼륨 체크 및 설정
print("\n=== Volume Check ===")
print(f"HF_HOME from env: {os.environ.get('HF_HOME', 'Not set')}")

# RunPod Serverless에서는 Network Volume이 /runpod-volume에 마운트됨
if os.path.exists("/runpod-volume"):
    stat = shutil.disk_usage("/runpod-volume")
    gb_free = stat.free / (1024**3)
    gb_total = stat.total / (1024**3)
    print(f"✅ Network volume found: {gb_free:.2f} GB free / {gb_total:.2f} GB total")
    
    # HF_HOME이 설정되어 있지 않으면 설정
    if not os.environ.get('HF_HOME'):
        os.environ['HF_HOME'] = "/runpod-volume/hf_cache"
        print(f"Setting HF_HOME to: {os.environ['HF_HOME']}")
else:
    print("❌ ERROR: /runpod-volume not found!")
    print("Available directories:")
    os.system("ls -la /")
    print("\nDisk usage:")
    os.system("df -h")
    print("\nMounted volumes:")
    os.system("mount | grep -E 'runpod|volume'")
    raise ValueError("Network volume not properly mounted!")

# 캐시 디렉토리 확인 및 생성
CACHE_DIR = os.environ.get('HF_HOME', '/runpod-volume/hf_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"Cache directory: {CACHE_DIR}")

# 전역 변수로 모델과 토크나이저 저장
model = None
tokenizer = None

def load_model():
    """모델과 토크나이저를 로드하는 함수"""
    global model, tokenizer
    
    if model is None:
        try:
            print("\n=== Model Loading ===")
            
            # 볼륨 공간 재확인
            if os.path.exists("/runpod-volume"):
                stat = shutil.disk_usage("/runpod-volume")
                gb_free = stat.free / (1024**3)
                print(f"Network volume free space: {gb_free:.2f} GB")
                if gb_free < 20:
                    print(f"⚠️ WARNING: Only {gb_free:.2f} GB free, need 20GB for model")
            
            print(f"GPU Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU Name: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            # bitsandbytes 호환성 설정
            if not hasattr(bnb.nn.Linear4bit, "ipex_linear_is_set"):
                bnb.nn.Linear4bit.ipex_linear_is_set = False
            
            # 모델 파일 경로
            model_filename = "model_complete.pt"
            local_model_path = os.path.join(CACHE_DIR, model_filename)
            
            # 이미 다운로드된 모델이 있는지 확인
            if os.path.exists(local_model_path):
                file_size_gb = os.path.getsize(local_model_path) / (1024**3)
                print(f"Found existing model: {local_model_path} ({file_size_gb:.2f} GB)")
                if file_size_gb > 10:  # 10GB 이상이면 유효한 모델로 간주
                    model_path = local_model_path
                else:
                    print("Model file seems incomplete, re-downloading...")
                    os.remove(local_model_path)
                    model_path = hf_hub_download(
                        repo_id="thegreatgame/exaone-accounting-complete",
                        filename=model_filename,
                        token=HF_TOKEN,
                        cache_dir=CACHE_DIR
                    )
            else:
                print(f"Downloading model to network volume: {CACHE_DIR}")
                print("First download will take time (~18.6 GB)...")
                
                # 모델 다운로드
                model_path = hf_hub_download(
                    repo_id="thegreatgame/exaone-accounting-complete",
                    filename=model_filename,
                    token=HF_TOKEN,
                    cache_dir=CACHE_DIR
                )
                
                print(f"Model downloaded to: {model_path}")
            
            # GPU 사용 가능 여부 확인
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading model to device: {device}")
            
            # 메모리 효율적인 로딩
            torch.cuda.empty_cache()
            
            # 파인튩닝된 모델 로드 (.pt 파일)
            print(f"Loading fine-tuned model from: {model_path}")
            model = torch.load(
                model_path,
                map_location=device,
                weights_only=False
            )
            print("Fine-tuned model loaded successfully!")
            
            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(
                "LGAI-EXAONE/EXAONE-4.0-32B",
                trust_remote_code=True,
                token=HF_TOKEN
            )
            
            # 모델을 평가 모드로 설정
            model.eval()
            
            print("Model loaded successfully!")
            if torch.cuda.is_available():
                print(f"GPU Memory After Loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    return model, tokenizer

def generate_response(question, use_cot=True):
    """주어진 질문에 대한 응답 생성"""
    global model, tokenizer
    
    # 모델이 로드되지 않았다면 로드
    if model is None or tokenizer is None:
        load_model()
    
    # 프롬프트 템플릿 설정
    if use_cot:
        prompt = """아래 지시문과 입력이 주어져 있다.
주어진 작업을 올바르게 수행하는 응답을 작성해라.
답변하기 전에 질문을 신중히 검토하고, 논리적 · 정확한 응답을 위해 단계별 Chain-of-Thought를 먼저 작성하라.

### Instruction:
너는 회계 전문가다. 제공된 거래 정보를 분석해 차변·대변을 정확히 분개하라.

### Question:
{}

### Response:
<think>""".format(question)
    else:
        prompt = """아래 지시문과 입력이 주어져 있다.
주어진 작업을 올바르게 수행하는 응답을 작성해라.

### Instruction:
너는 회계 전문가다. 제공된 거래 정보를 분석해 차변·대변을 정확히 분개하라.

### Question:
{}

### Response:
""".format(question)
    
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 입력 토큰화
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 텍스트 생성
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1200,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    # 응답 디코딩
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # 응답에서 필요한 부분만 추출
    try:
        result = response[0].split("### Response:")[1].strip()
    except:
        result = response[0]
    
    return result

def handler(event):
    """
    RunPod 서버리스 핸들러 함수
    
    Args:
        event (dict): RunPod 이벤트 객체
            - input: 입력 데이터를 포함하는 딕셔너리
                - question: 회계 관련 질문
                - use_cot: Chain-of-Thought 사용 여부 (기본값: True)
    
    Returns:
        dict: 응답 결과
            - output: 생성된 응답 텍스트
            - status: 처리 상태
    """
    try:
        # 입력 데이터 추출
        input_data = event.get("input", {})
        question = input_data.get("question", "")
        use_cot = input_data.get("use_cot", True)
        
        # 질문이 없으면 에러 반환
        if not question:
            return {
                "error": "Question is required",
                "status": "error"
            }
        
        # 응답 생성
        response = generate_response(question, use_cot)
        
        return {
            "output": response,
            "status": "success"
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

# RunPod 서버리스 엔트리포인트
if __name__ == "__main__":
    try:
        # 모델 미리 로드 (콜드 스타트 시간 단축)
        print("Preloading model...")
        load_model()
        print("Model preloading completed successfully")
        
        # RunPod 핸들러 시작
        print("Starting RunPod handler...")
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"Fatal error during initialization: {str(e)}")
        import sys
        sys.exit(1)