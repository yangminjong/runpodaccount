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

# RMSNorm 클래스 (많은 LLM에서 사용)
class Exaone4RMSNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        return x

# Attention 클래스
class Exaone4Attention(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.q_proj = nn.Linear(1, 1)
        self.k_proj = nn.Linear(1, 1)
        self.v_proj = nn.Linear(1, 1)
        self.o_proj = nn.Linear(1, 1)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is a dummy class for pickle loading")

# MLP 클래스
class Exaone4MLP(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.gate_proj = nn.Linear(1, 1)
        self.up_proj = nn.Linear(1, 1)
        self.down_proj = nn.Linear(1, 1)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is a dummy class for pickle loading")

# DecoderLayer 클래스
class Exaone4DecoderLayer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.self_attn = Exaone4Attention(config)
        self.mlp = Exaone4MLP(config)
        self.input_layernorm = Exaone4RMSNorm()
        self.post_attention_layernorm = Exaone4RMSNorm()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is a dummy class for pickle loading")

# RotaryEmbedding 클래스
class Exaone4RotaryEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is a dummy class for pickle loading")

class Exaone4Model(PreTrainedModel):
    config_class = Exaone4Config
    
    def __init__(self, config=None):
        if config is None:
            config = Exaone4Config()
        super().__init__(config)
        self.layers = nn.ModuleList([Exaone4DecoderLayer(config)])
        self.embed_tokens = nn.Embedding(1, 1)
        self.norm = Exaone4RMSNorm()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is a dummy class for pickle loading")

class Exaone4ForCausalLM(PreTrainedModel):
    config_class = Exaone4Config
    
    def __init__(self, config=None):
        if config is None:
            config = Exaone4Config()
        super().__init__(config)
        self.model = Exaone4Model(config)
        self.lm_head = nn.Module()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is a dummy class for pickle loading")
    
    def generate(self, *args, **kwargs):
        raise NotImplementedError("This is a dummy class for pickle loading")

# 모듈 생성
exaone4_module = types.ModuleType('transformers.models.exaone4')
modeling_module = types.ModuleType('transformers.models.exaone4.modeling_exaone4')

# 클래스 할당 (다양한 이름 형식 지원)
# 모든 클래스를 modeling_module에 등록
modeling_module.Exaone4Config = Exaone4Config
modeling_module.Exaone4RMSNorm = Exaone4RMSNorm
modeling_module.Exaone4Attention = Exaone4Attention
modeling_module.Exaone4MLP = Exaone4MLP
modeling_module.Exaone4DecoderLayer = Exaone4DecoderLayer
modeling_module.Exaone4RotaryEmbedding = Exaone4RotaryEmbedding
modeling_module.Exaone4Model = Exaone4Model
modeling_module.Exaone4ForCausalLM = Exaone4ForCausalLM

# 대문자 버전도 추가 (호환성)
modeling_module.EXAONE4Config = Exaone4Config
modeling_module.EXAONE4RMSNorm = Exaone4RMSNorm
modeling_module.EXAONE4Attention = Exaone4Attention
modeling_module.EXAONE4MLP = Exaone4MLP
modeling_module.EXAONE4DecoderLayer = Exaone4DecoderLayer
modeling_module.EXAONE4RotaryEmbedding = Exaone4RotaryEmbedding
modeling_module.EXAONE4Model = Exaone4Model
modeling_module.EXAONE4ForCausalLM = Exaone4ForCausalLM

# exaone4_module에도 등록
exaone4_module.modeling_exaone4 = modeling_module
exaone4_module.Exaone4Config = Exaone4Config
exaone4_module.Exaone4RMSNorm = Exaone4RMSNorm
exaone4_module.Exaone4Attention = Exaone4Attention
exaone4_module.Exaone4MLP = Exaone4MLP
exaone4_module.Exaone4DecoderLayer = Exaone4DecoderLayer
exaone4_module.Exaone4RotaryEmbedding = Exaone4RotaryEmbedding
exaone4_module.Exaone4Model = Exaone4Model
exaone4_module.Exaone4ForCausalLM = Exaone4ForCausalLM

# 시스템 모듈에 등록
sys.modules['transformers.models.exaone4'] = exaone4_module
sys.modules['transformers.models.exaone4.modeling_exaone4'] = modeling_module

print("EXAONE4 module loaded with actual classes for pickle compatibility")