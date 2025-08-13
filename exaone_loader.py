"""
EXAONE 모델 클래스 정의 - pickle 로딩을 위한 더미 클래스
"""
import sys
from unittest.mock import MagicMock

# transformers.models.exaone4 모듈을 모킹
sys.modules['transformers.models.exaone4'] = MagicMock()
sys.modules['transformers.models.exaone4.modeling_exaone4'] = MagicMock()

print("EXAONE4 module mocked for loading fine-tuned model")