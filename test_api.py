"""
RunPod API 테스트 스크립트
"""
import requests
import json
import os
from typing import Dict, Any

def test_runpod_api(
    endpoint_id: str,
    api_key: str,
    question: str,
    use_cot: bool = True
) -> Dict[str, Any]:
    """
    RunPod API를 테스트합니다.
    
    Args:
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
        question: 회계 관련 질문
        use_cot: Chain-of-Thought 사용 여부
    
    Returns:
        API 응답
    """
    
    # API URL 구성
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    
    # 헤더 설정
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 요청 데이터
    data = {
        "input": {
            "question": question,
            "use_cot": use_cot,
            "max_new_tokens": 1200,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    
    print("=" * 50)
    print("RunPod API 테스트")
    print("=" * 50)
    print(f"Endpoint: {endpoint_id}")
    print(f"Use CoT: {use_cot}")
    print(f"Question: {question[:100]}...")
    print("-" * 50)
    
    try:
        # API 호출
        print("API 호출 중...")
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        # 결과 출력
        if result.get("status") == "COMPLETED":
            print("✅ 성공!")
            print("-" * 50)
            print("응답:")
            output = result.get("output", {})
            if isinstance(output, dict):
                print(output.get("response", "응답 없음"))
            else:
                print(output)
        else:
            print("⚠️ 처리 중 또는 실패")
            print(f"상태: {result.get('status')}")
            print(f"상세: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API 호출 실패: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return {"error": str(e)}

def main():
    """메인 테스트 함수"""
    
    # 환경 변수에서 설정 읽기
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
    api_key = os.getenv("RUNPOD_API_KEY")
    
    if not endpoint_id or not api_key:
        print("❌ 환경 변수를 설정해주세요:")
        print("  export RUNPOD_ENDPOINT_ID=your_endpoint_id")
        print("  export RUNPOD_API_KEY=your_api_key")
        return
    
    # 테스트 케이스 1: Chain-of-Thought 사용
    test_question_1 = """회사명: 케이이노텍, 장표구분: 매입매출전표, 거래일자: 2022/01/03, 거래처: sk브로드밴드, 거래구분: 매입, 거래유형: 과세, 금액: 33,680원, 부가세: 3,368원, 합계: 37,048원, 결제방법: 외상, 적요: 1월 데이터요금"""
    
    print("\n테스트 1: Chain-of-Thought 사용")
    result1 = test_runpod_api(endpoint_id, api_key, test_question_1, use_cot=True)
    
    # 테스트 케이스 2: Chain-of-Thought 미사용
    test_question_2 = """회사명: 워터이지텍, 장표구분: 매입매출전표, 거래일자: 2022/04/08, 거래처: (주)건구종합건설, 거래구분: 환입, 거래유형: 과세, 금액: -660,000원, 부가세: -66,000원, 합계: -726,000원, 결제방법: 예금, 적요: 22-38번 계약 부분취소(차액환급)"""
    
    print("\n테스트 2: Chain-of-Thought 미사용")
    result2 = test_runpod_api(endpoint_id, api_key, test_question_2, use_cot=False)
    
    print("\n" + "=" * 50)
    print("테스트 완료!")
    
if __name__ == "__main__":
    main()