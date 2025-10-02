
from typing import Optional

import torch
from torch import nn
from torch.export import Dim
from test.utils.tag import use_onert


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states



@use_onert
class Lfm2ShortConv(nn.Module):
    # 분기 1: past_key_value 있고 cache_position[0] > 0 일 때 처리용 모듈
    class WithCache(nn.Module):
        """
        One-by-one Decoding Stage
        
        Condition; past_key_value (conv_cache) exists and cache_position[0] > 0 
        """
        def __init__(self, conv, layer_idx, L_cache, bias):
            super().__init__()
            self.conv = conv
            self.layer_idx = layer_idx
            self.L_cache = L_cache
            self.bias = bias

        def forward(self, Bx, cache_position, seq_len, conv_state):
            cache_position = cache_position.clamp(0, self.L_cache - 1)
            new_conv_state = conv_state.roll(shifts=-1, dims=-1)
            new_conv_state[:, :, cache_position] = Bx.to(device=conv_state.device, dtype=conv_state.dtype)
            
            conv_out = torch.sum(conv_state.to(Bx.device) * self.conv.weight[:, 0, :], dim=-1)
            if self.bias:
                conv_out += self.conv.bias

            conv_out = conv_out.unsqueeze(-1)
            return conv_out #, new_conv_state

    # 분기 2: 그 외 케이스 처리 모듈
    class WithoutCache(nn.Module):
        """
        Pure convolution
        """
        def __init__(self, conv, layer_idx, L_cache):
            super().__init__()
            self.conv = conv
            self.layer_idx = layer_idx
            self.L_cache = L_cache

        def forward(self, Bx, cache_position, seqlen, conv_state):
            new_conv_state = nn.functional.pad(Bx, (self.L_cache - Bx.shape[-1], 0))
            FIXED_SEQ_LEN = 20 # instead of using seqlen
            conv_out = self.conv(Bx)[..., :FIXED_SEQ_LEN] # for compilation
            return conv_out #, new_conv_state
        
    def __init__(
        self,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.L_cache = 3                      
        self.bias = False                    

        self.conv = nn.Conv1d(
            in_channels=1024,                
            out_channels=1024,               
            kernel_size=self.L_cache,       
            groups=1024,                
            bias=self.bias,
            padding=self.L_cache - 1,  
        )
        self.in_proj = nn.Linear(
            1024,                    
            3 * 1024,                   
            bias=self.bias,
        )
        self.out_proj = nn.Linear(
            1024,                       
            1024,
            bias=self.bias,
        )
        # # Sharing self.conv is not supported by torch.export
        # self.conv_with_cache = self.WithCache(self.conv, self.layer_idx, self.L_cache, self.bias)
        # self.conv_without_cache = self.WithoutCache(self.conv, self.layer_idx, self.L_cache)
        self.conv_with_cache = self.WithCache(nn.Conv1d(
            in_channels=1024,                
            out_channels=1024,               
            kernel_size=self.L_cache,       
            groups=1024,                
            bias=self.bias,
            padding=self.L_cache - 1,  
        ), self.layer_idx, self.L_cache, self.bias)
        self.conv_without_cache = self.WithoutCache(nn.Conv1d(
            in_channels=1024,                
            out_channels=1024,               
            kernel_size=self.L_cache,       
            groups=1024,                
            bias=self.bias,
            padding=self.L_cache - 1,  
        ), self.layer_idx, self.L_cache)

    # def slow_forward(
    def forward(
        self,
        x: torch.Tensor,
        conv_cache = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        seqlen = torch.tensor(x.shape[1])

        x = apply_mask_to_padding_states(x, attention_mask)
        BCx = self.in_proj(x).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)

        Bx = B * x
        
        condition = conv_cache is not None and cache_position[0] > 0
        
        conv_state = conv_cache[self.layer_idx]
        
        # # TODO Enable cache state update after exportation
        # conv_out, new_conv_state = torch.cond(condition, 
        conv_out = torch.cond(condition, 
                   self.conv_with_cache, self.conv_without_cache,
                   (Bx, cache_position, seqlen, conv_state,))
        # conv_cache[self.layer_idx, :, :, :] = new_conv_state.clone()
        
        y = C * conv_out
        y = y.transpose(-1, -2).contiguous()
        y = self.out_proj(y)
        return y
    
    
    def get_example_inputs(self):
        sequence_length = 1 
        x = torch.randn(32, sequence_length, 1024)
        max_batch_size = 32
        conv_dim = 1024
        conv_L_cache = 3
        num_hidden_layers = 12
        
        conv_cache = torch.zeros(num_hidden_layers, max_batch_size, conv_dim, conv_L_cache, dtype=torch.float32)
        cache_position = torch.tensor([5])
        
        # assert (conv_cache is not None and sequence_length == 1) or (conv_cache is None)
        
        return (x, conv_cache, cache_position,), {}

    def get_dynamic_shapes(self):
        sequence_length = Dim("sequence_length", min=1, max=128)
        dynamic_shapes = {
            "x": {1: sequence_length},
            "conv_cache": {},
            "cache_position": {},
        }

        return dynamic_shapes
