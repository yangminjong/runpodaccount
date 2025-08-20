"""
EXAONE 모델 클래스 정의 - pickle 로딩을 위한 실제 클래스
"""
import sys
import types
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class Exaone4Config(PretrainedConfig):
    model_type = "exaone4"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

class Exaone4ForCausalLM(PreTrainedModel):
    config_class = Exaone4Config
    
    def __init__(self, config=None):
        if config is None:
            config = Exaone4Config()
        super().__init__(config)
        self.model = nn.Module()
        self.lm_head = nn.Module()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is a dummy class for pickle loading")
    
    def generate(self, *args, **kwargs):
        raise NotImplementedError("This is a dummy class for pickle loading")

# 모듈 생성
exaone4_module = types.ModuleType('transformers.models.exaone4')
modeling_module = types.ModuleType('transformers.models.exaone4.modeling_exaone4')

# 클래스 할당 (다양한 이름 형식 지원)
modeling_module.Exaone4ForCausalLM = Exaone4ForCausalLM
modeling_module.Exaone4Config = Exaone4Config
modeling_module.EXAONE4ForCausalLM = Exaone4ForCausalLM  # 대문자 버전도 추가
modeling_module.EXAONE4Config = Exaone4Config  # 대문자 버전도 추가
exaone4_module.modeling_exaone4 = modeling_module
exaone4_module.Exaone4ForCausalLM = Exaone4ForCausalLM
exaone4_module.Exaone4Config = Exaone4Config
exaone4_module.EXAONE4ForCausalLM = Exaone4ForCausalLM  # 대문자 버전도 추가
exaone4_module.EXAONE4Config = Exaone4Config  # 대문자 버전도 추가

# 시스템 모듈에 등록
sys.modules['transformers.models.exaone4'] = exaone4_module
sys.modules['transformers.models.exaone4.modeling_exaone4'] = modeling_module

print("EXAONE4 module loaded with actual classes for pickle compatibility")