"""
RunPod Serverless Handler for Fine-tuned EXAONE Model
"""
import runpod
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import bitsandbytes as bnb
import os

# BitsAndBytes 설정
if not hasattr(bnb.nn.Linear4bit, "ipex_linear_is_set"):
    bnb.nn.Linear4bit.ipex_linear_is_set = False

# 전역 변수로 모델과 토크나이저 선언
model = None
tokenizer = None

def check_storage():
    """스토리지 용량을 체크합니다."""
    import shutil
    storage_path = "/runpod-volume"
    
    if os.path.exists(storage_path):
        total, used, free = shutil.disk_usage(storage_path)
        print(f"Storage info for {storage_path}:")
        print(f"  Total: {total // (2**30)} GB")
        print(f"  Used: {used // (2**30)} GB")
        print(f"  Free: {free // (2**30)} GB")
        
        # 모델 크기가 약 18.6GB이므로 최소 20GB의 여유 공간 필요
        if free < 20 * (2**30):
            print(f"WARNING: Low storage space! Free: {free // (2**30)} GB")
    else:
        print(f"Storage path {storage_path} not found, using local storage")

def load_model():
    """모델과 토크나이저를 로드합니다."""
    global model, tokenizer
    
    # 스토리지 체크
    check_storage()
    
    # Hugging Face 환경 변수 설정
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")
    
    # HF_HOME 설정 (RunPod 스토리지 사용)
    hf_home = os.environ.get("HF_HOME", "/runpod-volume/hf_cache")
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_home
    
    # 캐시 디렉토리 생성
    os.makedirs(hf_home, exist_ok=True)
    
    print(f"Using HF_HOME: {hf_home}")
    print(f"Region: US-WA-1 (US West)")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "LGAI-EXAONE/EXAONE-4.0-32B",
        trust_remote_code=True,
        token=hf_token,
        cache_dir=hf_home
    )
    
    print("Downloading and loading model...")
    model_path = hf_hub_download(
        repo_id="thegreatgame/exaone-accounting-complete",
        filename="model_complete.pt",
        token=hf_token,
        cache_dir=hf_home
    )
    
    # GPU 사용 가능 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )
    
    # 모델을 명시적으로 GPU로 이동
    if device == "cuda":
        model = model.cuda()
        print("Model moved to GPU")
    
    model.eval()  # 추론 모드로 설정
    print("Model loaded successfully!")
    
    # 모델 로드 후 스토리지 재확인
    check_storage()

def handler(job):
    """
    RunPod serverless handler 함수
    
    Args:
        job: RunPod job dictionary with input data
        
    Returns:
        Response dictionary
    """
    global model, tokenizer
    
    # 디버깅을 위한 로그
    print(f"Received job: {job}")
    
    # 모델이 로드되지 않았다면 로드
    if model is None or tokenizer is None:
        load_model()
    
    # 입력 데이터 가져오기
    job_input = job.get("input", {})
    print(f"Job input: {job_input}")
    
    # 필수 입력 검증
    question = job_input.get("question")
    if not question:
        print(f"ERROR: Question not found in input. Available keys: {list(job_input.keys())}")
        return {"error": "Question is required"}
    
    # 선택적 파라미터
    use_cot = job_input.get("use_cot", True)  # Chain-of-Thought 사용 여부
    max_new_tokens = job_input.get("max_new_tokens", 1200)
    temperature = job_input.get("temperature", 0.7)
    top_p = job_input.get("top_p", 0.9)
    
    # 프롬프트 생성
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
    
    try:
        # 토크나이징
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # GPU로 이동
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 추론 실행
        print(f"Starting inference with max_new_tokens={max_new_tokens}")
        with torch.no_grad():
            # temperature와 top_p를 사용하려면 do_sample=True 필요
            if temperature != 1.0 or top_p != 1.0:
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            else:
                # 기본 생성 (더 빠름)
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
        print("Inference completed")
        
        # 디코딩
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Response 부분만 추출
        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()
        
        # <｜end▁of▁sentence｜> 토큰 제거
        response = response.replace("<｜end▁of▁sentence｜>", "").strip()
        
        return {
            "response": response,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

# RunPod serverless 실행
print("Loading RunPod handler module...")

# 모델 사전 로드 (선택사항 - 콜드 스타트 방지)
try:
    print("Pre-loading model for faster inference...")
    load_model()
except Exception as e:
    print(f"Warning: Could not pre-load model: {e}")

if __name__ == "__main__":
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})