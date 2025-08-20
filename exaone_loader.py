"""
EXAONE 모델 클래스 정의 - pickle 로딩을 위한 실제 클래스
"""
import sys
import types
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

# RoPE 유틸리티 모듈 생성
rope_utils_module = types.ModuleType('transformers.modeling_rope_utils')

# RoPE 관련 함수들 (더미 구현)
def apply_rotary_pos_emb(*args, **kwargs):
    return args[0] if args else None

def rotate_half(*args, **kwargs):
    return args[0] if args else None

# 모듈에 함수 할당
rope_utils_module.apply_rotary_pos_emb = apply_rotary_pos_emb
rope_utils_module.rotate_half = rotate_half

# 시스템 모듈에 등록
sys.modules['transformers.modeling_rope_utils'] = rope_utils_module

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
        return None  # 더미 구현

# MLP 클래스
class Exaone4MLP(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.gate_proj = nn.Linear(1, 1)
        self.up_proj = nn.Linear(1, 1)
        self.down_proj = nn.Linear(1, 1)
    
    def forward(self, *args, **kwargs):
        return None  # 더미 구현

# DecoderLayer 클래스
class Exaone4DecoderLayer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.self_attn = Exaone4Attention(config)
        self.mlp = Exaone4MLP(config)
        self.input_layernorm = Exaone4RMSNorm()
        self.post_attention_layernorm = Exaone4RMSNorm()
    
    def forward(self, *args, **kwargs):
        return None  # 더미 구현

# RotaryEmbedding 클래스
class Exaone4RotaryEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        return None  # 더미 구현

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
        return None  # 더미 구현

# Tokenizer 관련 클래스들
class Exaone4Tokenizer:
    def __init__(self, *args, **kwargs):
        pass

class Exaone4TokenizerFast:
    def __init__(self, *args, **kwargs):
        pass

class Exaone4ForCausalLM(PreTrainedModel):
    config_class = Exaone4Config
    
    def __init__(self, config=None):
        if config is None:
            config = Exaone4Config()
        super().__init__(config)
        self.model = Exaone4Model(config)
        self.lm_head = nn.Module()
    
    def forward(self, *args, **kwargs):
        # pickle로 로드된 실제 모델이 이 메서드를 덮어씀
        return super().forward(*args, **kwargs) if hasattr(super(), 'forward') else None
    
    def generate(self, *args, **kwargs):
        # PreTrainedModel의 generate 메서드 사용
        return super().generate(*args, **kwargs)

# 모듈 생성 - exaone4를 패키지처럼 만들기
exaone4_module = types.ModuleType('transformers.models.exaone4')
exaone4_module.__path__ = []  # 패키지로 만들기 위해 __path__ 추가
exaone4_module.__file__ = 'transformers/models/exaone4/__init__.py'

# 서브모듈들 생성
modeling_module = types.ModuleType('transformers.models.exaone4.modeling_exaone4')
configuration_module = types.ModuleType('transformers.models.exaone4.configuration_exaone4')
tokenization_module = types.ModuleType('transformers.models.exaone4.tokenization_exaone4')
tokenization_fast_module = types.ModuleType('transformers.models.exaone4.tokenization_exaone4_fast')

# __init__ 모듈도 생성
init_module = types.ModuleType('transformers.models.exaone4.__init__')

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

# configuration_module에 Config 클래스 등록
configuration_module.Exaone4Config = Exaone4Config
configuration_module.EXAONE4Config = Exaone4Config

# tokenization 모듈에 토크나이저 클래스 등록
tokenization_module.Exaone4Tokenizer = Exaone4Tokenizer
tokenization_module.EXAONE4Tokenizer = Exaone4Tokenizer
tokenization_fast_module.Exaone4TokenizerFast = Exaone4TokenizerFast
tokenization_fast_module.EXAONE4TokenizerFast = Exaone4TokenizerFast

# init 모듈에 모든 클래스 등록
init_module.Exaone4Config = Exaone4Config
init_module.Exaone4Model = Exaone4Model
init_module.Exaone4ForCausalLM = Exaone4ForCausalLM
init_module.Exaone4Tokenizer = Exaone4Tokenizer
init_module.Exaone4TokenizerFast = Exaone4TokenizerFast

# exaone4_module에도 등록
exaone4_module.modeling_exaone4 = modeling_module
exaone4_module.configuration_exaone4 = configuration_module
exaone4_module.tokenization_exaone4 = tokenization_module
exaone4_module.tokenization_exaone4_fast = tokenization_fast_module
exaone4_module.__init__ = init_module
exaone4_module.Exaone4Config = Exaone4Config
exaone4_module.Exaone4RMSNorm = Exaone4RMSNorm
exaone4_module.Exaone4Attention = Exaone4Attention
exaone4_module.Exaone4MLP = Exaone4MLP
exaone4_module.Exaone4DecoderLayer = Exaone4DecoderLayer
exaone4_module.Exaone4RotaryEmbedding = Exaone4RotaryEmbedding
exaone4_module.Exaone4Model = Exaone4Model
exaone4_module.Exaone4ForCausalLM = Exaone4ForCausalLM
exaone4_module.Exaone4Tokenizer = Exaone4Tokenizer
exaone4_module.Exaone4TokenizerFast = Exaone4TokenizerFast

# 시스템 모듈에 등록
sys.modules['transformers.models.exaone4'] = exaone4_module
sys.modules['transformers.models.exaone4.modeling_exaone4'] = modeling_module
sys.modules['transformers.models.exaone4.configuration_exaone4'] = configuration_module
sys.modules['transformers.models.exaone4.tokenization_exaone4'] = tokenization_module
sys.modules['transformers.models.exaone4.tokenization_exaone4_fast'] = tokenization_fast_module
sys.modules['transformers.models.exaone4.__init__'] = init_module

print("EXAONE4 module loaded with actual classes for pickle compatibility")