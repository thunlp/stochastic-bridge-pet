import torch
import torch.nn.functional as F
from functools import partial

from megatron import get_timers, get_args
from megatron import mpu
from .module import MegatronModule
from megatron.model.enums import AttnMaskType, ModelType, LayerType, AttnType
from megatron.model import LayerNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu

class LoRA(MegatronModule):
    def __init__(self, hidden_size, r=8, alpha=8):
        super().__init__()
        self.down_project = mpu.ColumnParallelLinear(hidden_size, r, gather_output=False, init_method=partial(torch.nn.init.normal_, std=0.02))
        self.up_project = mpu.RowParallelLinear(r, hidden_size, input_is_parallel=True, init_method=torch.nn.init.zeros_)
        self.alpha = alpha / r

    def forward(self, hidden_states):
        seq_len, bsz, np, hn = hidden_states.size()
        hidden_states = hidden_states.view(seq_len, bsz, np * hn)
        hidden_states = self.alpha * self.up_project(self.down_project(hidden_states))
        return hidden_states.view(seq_len, bsz, np, hn)
