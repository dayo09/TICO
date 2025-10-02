import torch
from torch import nn
from torch.export import export  # PyTorch 2.x에서 사용


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states

# 분기 1: past_key_value 있고 cache_position[0] > 0 일 때 처리용 모듈
class TrueBranch(nn.Module):
    def __init__(self, conv, conv_cache, layer_idx, L_cache, bias):
        super().__init__()
        self.conv = conv
        self.register_buffer(
            "conv_cache",
            torch.zeros(10, 32, 1024, 20),
        )
        self.layer_idx = layer_idx
        self.L_cache = L_cache
        self.bias = bias

    def forward(self, Bx, cache_position, seq_len):
        conv_state = self.conv_cache[self.layer_idx,:,:,:]
        cache_position = torch.clamp(cache_position, 0, self.L_cache - 1)
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = Bx.to(device=conv_state.device, dtype=conv_state.dtype)
        self.conv_cache[self.layer_idx, :, :, :] = conv_state.clone()
        conv_out = torch.sum(conv_state.to(Bx.device) * self.conv.weight[:, 0, :], dim=-1)
        if self.bias is not None:
            conv_out += self.bias
        conv_out = conv_out.unsqueeze(-1)
        return conv_out

# 분기 2: 그 외 케이스 처리 모듈
class FalseBranch(nn.Module):
    def __init__(self, conv, conv_cache, layer_idx, L_cache):
        super().__init__()
        self.conv = conv
        self.register_buffer(
            "conv_cache",
            torch.zeros(10, 32, 1024, 20),
        )
        self.layer_idx = layer_idx
        self.L_cache = L_cache

    def forward(self, Bx, cache_position, seqlen):
        conv_state = nn.functional.pad(Bx, (self.L_cache - Bx.shape[-1], 0))
        self.conv_cache[self.layer_idx, :, :, :] = conv_state.clone()
        conv_out = self.conv(Bx)[..., :seqlen]
        return conv_out

class ShortConv(nn.Module):
    def __init__(self, in_proj, conv, conv_cache, out_proj, layer_idx, L_cache, bias):
        super().__init__()
        self.in_proj = in_proj
        self.conv = conv
        self.conv_cache = conv_cache
        self.out_proj = out_proj
        self.layer_idx = layer_idx
        self.L_cache = L_cache
        self.bias = bias

        self.true_branch = TrueBranch(conv, conv_cache, layer_idx, L_cache, bias)
        self.false_branch = FalseBranch(conv, conv_cache, layer_idx, L_cache)

    def forward(self, x, past_key_value=None, cache_position=None, attention_mask=None):
        seqlen = torch.tensor(x.shape[1])
        x = apply_mask_to_padding_states(x, attention_mask)
        BCx = self.in_proj(x).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)
        Bx = B * x


        # 조건: past_key_value가 있고 cache_position[0] > 0 인 경우
        pred = (past_key_value is not None) and (cache_position[0] > 0)
        # if pred is True:
        #     conv_out = true_fn()
        # else:
        #     conv_out = false_fn()
        conv_out = torch.cond(pred, self.true_branch, self.false_branch, (Bx, cache_position, seqlen,))

        y = C * conv_out
        y = y.transpose(-1, -2).contiguous()
        y = self.out_proj(y)
        return y

import torch
from torch import nn
from torch.export import export

# 앞서 정의한 TrueBranch, FalseBranch, MoEModel 클래스가 있다고 가정

# 모듈 인스턴스화 시 필요한 임의 파라미터 초기화 (예시)
in_proj = nn.Linear(1024, 1024 * 3)       # 임베딩 크기 1024 가정
conv = nn.Conv1d(1024, 1024, kernel_size=(3,), stride=(1,), padding=(2,), groups=1024, bias=False) # 1D conv, feature 채널 64
conv_cache = [torch.zeros(32, 1024, 20)] * 10         # 배치 32, feature 64, 캐시 크기 20, 레이어 10개 가정
out_proj = nn.Linear(1024, 1024)            # 출력 임베딩 크기 1024

layer_idx = 0
L_cache = 20
bias = conv.bias

# MoEModel 생성
model = MoEModel(in_proj, conv, conv_cache, out_proj, layer_idx, L_cache, bias)

# 예시 입력 생성 (배치 32, 시퀀스 길이 10, 임베딩 1024)
x = torch.randn(32, 10, 1024)

# past_key_value와 cache_position도 필요한 경우 생성 (None 가능)
past_key_value = type('', (), {})()
past_key_value.conv_cache = conv_cache
cache_position = torch.tensor([5])

model.forward(x, past_key_value, cache_position, None) #ADDED
# model.eval()
# # torch.export로 모델 export 예시
# exported_model = export(model, (x, past_key_value.conv_cache, cache_position, None)) 

# # ExportedProgram 타입 출력 확인
# print(type(exported_model))
# print(exported_model)

# import tico

# tico.convert_from_exported_program(exported_model).save("shortconv.circle")
