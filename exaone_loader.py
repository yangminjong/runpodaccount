"""
EXAONE 모델 클래스 정의 - pickle 로딩을 위한 실제 클래스
"""
import sys
import types
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class EXAONE4Config(PretrainedConfig):
    model_type = "exaone4"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

class EXAONE4ForCausalLM(PreTrainedModel):
    config_class = EXAONE4Config
    
    def __init__(self, config=None):
        if config is None:
            config = EXAONE4Config()
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

# 클래스 할당
modeling_module.EXAONE4ForCausalLM = EXAONE4ForCausalLM
modeling_module.EXAONE4Config = EXAONE4Config
exaone4_module.modeling_exaone4 = modeling_module
exaone4_module.EXAONE4ForCausalLM = EXAONE4ForCausalLM
exaone4_module.EXAONE4Config = EXAONE4Config

# 시스템 모듈에 등록
sys.modules['transformers.models.exaone4'] = exaone4_module
sys.modules['transformers.models.exaone4.modeling_exaone4'] = modeling_module

print("EXAONE4 module loaded with actual classes for pickle compatibility")