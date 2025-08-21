"""
RunPod API 간단한 테스트
"""
import requests
import json

# 설정 (실제 값으로 변경 필요)
ENDPOINT_ID = "your_endpoint_id_here"  # 예: "bg70vb3rd4piyd"
API_KEY = "your_api_key_here"

# API URL
url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

# 헤더
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 요청 데이터 - input 안에 question이 있어야 함!
data = {
    "input": {
        "question": "회사명: 케이이노텍, 장표구분: 매입매출전표, 거래일자: 2022/01/03, 거래처: sk브로드밴드, 거래구분: 매입, 거래유형: 과세, 금액: 33,680원, 부가세: 3,368원, 합계: 37,048원, 결제방법: 외상, 적요: 1월 데이터요금",
        "use_cot": True,
        "max_new_tokens": 1200
    }
}

print("Sending request to RunPod...")
print(f"URL: {url}")
print(f"Data: {json.dumps(data, ensure_ascii=False, indent=2)}")

try:
    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    
    print("\n" + "="*50)
    print("Response:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    if result.get("status") == "COMPLETED":
        output = result.get("output", {})
        if isinstance(output, dict) and "response" in output:
            print("\n회계 분개 결과:")
            print(output["response"])
    
except Exception as e:
    print(f"Error: {e}")