# Standard library imports
import inspect
import json
import math
import os
import re
import sys
import traceback
import types
from typing import Dict, List, Optional, Tuple, Union, Any

# PyTorch and related imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# PyTorch Geometric imports
from torch_geometric.nn import GATConv, HANConv, HGTConv, HeteroLayerNorm, LayerNorm, SAGPooling
from torch_geometric.utils import dense_to_sparse, to_dense_batch, unbatch_edge_index

# Transformers core imports
from transformers import AutoProcessor, PreTrainedModel, Qwen3VLForConditionalGeneration, Qwen3VLPreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    ModelOutput
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# Model-specific imports
from transformers.models.mistral.modeling_mistral import create_sliding_window_causal_mask
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModelOutputWithPast,
    Qwen3VLCausalLMOutputWithPast,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

# Accelerate import
from accelerate import init_empty_weights

# PEFT import
from peft import PeftModel

# Local project imports
from lora.GNN import RGATConv, SRGATConv
from lora.Router import LowRankAdapterRouter, RouterManager
from lora.config import LoRAConfig, PretrainedConfig, TARGET_MODULE_TYPE, TKGAdapterConfig
from dataclasses import dataclass


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Qwen3VLConfig"

class LoRaModel:
    @staticmethod
    def from_pretrained(
        model: PreTrainedModel,
        name_or_path: Optional[str] = None,
    ) -> PeftModel:
        with open(name_or_path + "config.json") as f:
            config = json.load(f)
        config = LoRAConfig.from_config(config)
        config.torch_dtype = model.dtype
        model = _apply_lora(model, config=config)
        return model

class LoRA(nn.Module):
    def __init__(self, base_layer: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.out_features, self.in_features = base_layer.weight.shape
        self.dtype_ = config.torch_dtype
        self.dropout_tate = config.dropout
        self.dropout = nn.Dropout(config.dropout)
        self.rank = config.lora_r
        self.lora_alpha = config.lora_alpha
        self.scaling = self.lora_alpha / self.rank
        self.rank = config.lora_r
        self.lora_a = nn.Parameter(
            torch.empty((self.rank, self.in_features), dtype=self.dtype_))
        self.lora_b = nn.Parameter(
            torch.empty((self.out_features, self.rank), dtype=self.dtype_))
        self.reset_parameters() # initialize the A and B matrices

    # initialize the A and B matrices
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = F.linear(hidden_states, self.lora_a)
        hidden_states = F.linear(hidden_states, self.lora_b) * self.scaling
        return hidden_states + residual

class AdapterLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear,
                 config: LoRAConfig,
                 router: None,
                 use_cache: bool = False,
                 use_lora: bool = True): # use lora
        super().__init__()
        # linear
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.weight = base_layer.weight
        if hasattr(base_layer, "bias"):
            self.bias = base_layer.bias
        else:
            self.register_parameter('bias', None)

        self.lora = LoRA(base_layer, config) # core LoRA module
       

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        result = F.linear(hidden_states, self.weight, self.bias)
        return self.lora(hidden_states, result)


def get_peft_model(model: PreTrainedModel, config: LoRAConfig) -> PeftModel:
    config.hidden_size = model.config.hidden_size
    config.model_type = model.config.model_type
    
    model = _apply_lora(model, config)
    return model


def _get_module(model: nn.Module, target_name: str):
    for name, module in model.named_modules():
        if name == target_name:
            return module
    return None


def _apply_for_layer(layer_module: nn.Module,
                     layer_id: int,
                     config: LoRAConfig):

    def get_target_modules(target: list[str]) -> list[str]:
        res = []
        for t in target:
            res += TARGET_MODULE_TYPE[config.model_type][t]
        return res

    def set_lora(module, targets: list):
        """
        :param module:
        :param targets:
        :return:
        """
        for target_name in targets: # inject LoRA into all target linear layers
            if target_name not in config.target_modules:
                continue
            target_model = _get_module(module, target_name)
            if not isinstance(target_model, nn.Linear): # check whether it is a linear layer (Q/K/V/O)
                continue
            # inject LoRA
            # if config.target_modules_lora is not None and target_name in config.target_modules_lora:
            if config.target_modules is not None and target_name in config.target_modules:
                target_model = AdapterLinear(target_model, config, router=None, use_cache=False, use_lora=True)
        
            # dynamically replace the model layer
            # NOTE: replace the target_name layer in the module with a LoRA layer
            setattr(module, target_name, target_model)

    # apply for attention block
    # NOTE: get the attention module
    atte_name = TARGET_MODULE_TYPE[config.model_type]['atte']
    atte_module = _get_module(layer_module, atte_name)

    target_modules = get_target_modules(['q', 'k', 'v', 'o'])
    for target_module in target_modules:
        set_lora(atte_module, [target_module])

   
    ffn_name = TARGET_MODULE_TYPE[config.model_type]['ffn']
    ffn_module = _get_module(layer_module, ffn_name)
    
    target_modules = get_target_modules(['wi', 'wo'])
    for target_module in target_modules:
        set_lora(ffn_module, [target_module])



def _apply_lora(model, config: LoRAConfig) -> PeftModel:

    def _extract_layer_id(name: str):
        """Precisely extract the Qwen language model decoder layer ID."""
        # The standard Qwen3VL language-model layer path contains 'layers.'
        # e.g. 'model.layers.0' or 'model.language_model.layers.0'
        match = re.search(r'layers\.(\d+)$', name)
        if match:
            return int(match.group(1))
        return None
    
    # Build the mapping from layer id to layer_module
    # NOTE: layer_list holds the existing module layers
    layer_list = dict()  # {layer_id : layer_module}
    for module_name, module in model.named_modules():
        layer_id = _extract_layer_id(module_name)
        if layer_id is not None:
            if layer_id > config.max_llm_layer:
                # record the max layer id
                config.max_llm_layer = layer_id
            # record decoder layers
            layer_list[layer_id] = module

    
    for layer_id in sorted(layer_list.keys()):
        module = layer_list[layer_id]
        _apply_for_layer(module, layer_id, config)

    trainable_modules = ['lora'] 
    
    for param_name, param in model.named_parameters():
        # enable gradients if the parameter name contains lora
        if any(target in param_name for target in trainable_modules):
            param.requires_grad = True

    # override save_pretrained with custom saving logic
    model.save_pretrained = types.MethodType(_save_pretrained, model)
    
    setattr(model, 'peft_config', config)
    
    return model

def _save_pretrained(self: nn.Module, path, state_dict=None):
    if not os.path.exists(path):
        os.makedirs(path)
    trainable_params = dict()
    
    # if an external full state_dict (with absolute keys) is passed, extract from it first
    if state_dict is not None:
        # collect the "relative" names of trainable params as a reference set
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        # iterate the absolute dict; save the absolute key when its suffix matches a trainable param
        for absolute_key, tensor in state_dict.items():
            if getattr(tensor, "requires_grad", False):
                trainable_params[absolute_key] = tensor.detach().cpu()
    else:
        # fall back to the relative-path strategy (if needed)
        for name, param in self.named_parameters():
            if param.requires_grad: 
                trainable_params[name] = param.detach().cpu()
                
    config = self.peft_config.export()
    
    # use the correct extension
    save_file_path = os.path.join(path, 'adapter_model.bin')
    torch.save(trainable_params, save_file_path) 
    config['torch_dtype'] = None
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed

class Qwen3RMSNorm(nn.Module):
    def __init__(self,hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class Qwen3RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen3MLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class KgAdapterMLP(nn.Module):
    # TODO: zero-init gate ?
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Lora_layer(nn.Module):
    def __init__(self, r, lora_alpha, in_features, out_features):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.scaling = self.lora_alpha / self.r

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, input_tensor):
        # input_tensor = self.lora_dropout(input_tensor)
        input_tensor = torch.matmul(input_tensor, self.lora_A)
        input_tensor = torch.matmul(input_tensor, self.lora_B)
        input_tensor = input_tensor * self.scaling

        return input_tensor

class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.add_lora = config.add_lora
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # handle the KV dim separately for the 3B model
        if "4B" in config.name_or_path:
            self.kv_heads = getattr(config, 'num_key_value_heads',
                getattr(getattr(config, 'text_config', config), 
                        'num_key_value_heads', self.num_heads))
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.kv_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        else:
            self.kv_heads = self.num_heads  # normal model: KV heads equal Q heads
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = Qwen3RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        
        if self.add_lora:
            self.lora_q = Lora_layer(r=16, lora_alpha=16 * 2, in_features=self.hidden_size,
                                     out_features=self.num_heads * self.head_dim)
            self.lora_v = Lora_layer(r=16, lora_alpha=16 * 2, in_features=self.hidden_size,
                                     out_features=self.num_heads * self.head_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()

        # projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.add_lora:
            query_states = query_states + self.lora_q(hidden_states)
            value_states = value_states + self.lora_v(hidden_states)

        # reshape tensors
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.kv_heads, self.head_dim).transpose(1, 2)

        # rotary position embedding
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # handle past_key_value
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # attention computation: handle GQA with different head counts
        if self.kv_heads != self.num_heads:
            # Grouped Query Attention: repeat KV heads to match Q heads
            key_states = key_states.repeat_interleave(self.num_heads // self.kv_heads, dim=1)
            value_states = value_states.repeat_interleave(self.num_heads // self.kv_heads, dim=1)

        # compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # attention mask
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # softmax and attention output
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # reshape the output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # output projection
        attn_output = self.o_proj(attn_output)

        # clear attention weights (if not needed)
        if not output_attentions:
            attn_weights = None

        # always return a 3-tuple
        return attn_output, attn_weights, past_key_value

class KgAdapterCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3VLConfig ):
        super().__init__()
        self.config = config
        self.hidden_size = config.kg_adapter_hidden_size
        self.num_heads = 4
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        if "4B" in config.name_or_path:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            q_hidden_states: torch.Tensor,
            k_hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, kv_len, hidden_size = q_hidden_states.size(0), q_hidden_states.size(1), k_hidden_states.size(
            1), q_hidden_states.size(-1)
        align_mask = None
        if isinstance(attention_mask, Tuple):
            align_mask = attention_mask[1]
            attention_mask = attention_mask[0]
        # q：node_rep, k,v: text_rep
        query_states = self.q_proj(q_hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(k_hidden_states).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(k_hidden_states).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if past_key_value[0].shape[2] != k_hidden_states.shape[1]:
                kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # reuse k, v, self_attention
            if past_key_value[0].shape[2] != k_hidden_states.shape[1]:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            else:
                key_states = past_key_value[0]
                value_states = past_key_value[1]

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        if align_mask is not None:
            align_mask = align_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).transpose(-1, -2)
            attn_weights = attn_weights.masked_fill(align_mask == 0, torch.finfo(attn_weights.dtype).min)
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class KgAdapterInfoMerge(nn.Module):
    def __init__(self, method, hidden_size):
        super().__init__()# nn.Module is the base class of all neural-network modules in PyTorch
        self.method = method
        if method == "gate":
            # initialized to a unit value; a suitable weight is learned during training
            # to control the fusion ratio between the main and side streams
            # self.side_gate_params = nn.Parameter(torch.full((1,), 4.0)) 
            self.side_gate_params = nn.Parameter(torch.zeros(1)) 
        elif method == "linear":
            self.proj = nn.Linear(hidden_size * 2, hidden_size)
        # else use sum

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor = None,
    ):
        if self.method == "gate":
            # gate = torch.sigmoid(self.side_gate_params)
            # x = gate * x1 + (1 - gate) * x2
            device = x1.device  # get the device of x1 (assume x1, x2 are on the same device)
            gate = torch.sigmoid(self.side_gate_params)  # make sure the gate is on the right device
            x2 = x2
            x = gate * x1 + (1 - gate) * x2

        elif self.method == "linear":
            if x3 is None:
                x = self.proj(torch.cat([x1, x2], dim=-1))
            else:
                x = self.proj(torch.cat([x1, x2, x3], dim=-1))
        else:
            if x3 is None:
                x = x1 + x2
            else:
                x = x1 + x2 + x3

        return x

class KgAdapterLayerInsertManager(nn.Module):
    def __init__(
        self,
        num_layers: int,
        target_active_layers: int = 8,
        candidate_layer_ids=None,
        init_prob: float = 0.5,
        init_noise_std: float = 0.2,
        use_hard_topk: bool = True,
    ):
        super().__init__()

        assert 0.0 < init_prob < 1.0

        base_logit = torch.log(
            torch.tensor(init_prob) / (1.0 - torch.tensor(init_prob))
        )

        # Adding noise is important; otherwise all layers are fully symmetric and learn identically
        gate_logits = base_logit + init_noise_std * torch.randn(num_layers)
        self.gate_logits = nn.Parameter(gate_logits)

        self.num_layers = num_layers
        self.target_active_layers = target_active_layers
        self.use_hard_topk = use_hard_topk

        if candidate_layer_ids is None:
            candidate_layer_ids = list(range(num_layers))

        candidate_mask = torch.zeros(num_layers, dtype=torch.bool)
        candidate_mask[candidate_layer_ids] = True
        self.register_buffer("candidate_mask", candidate_mask)

    def gate_probs(self):
        return torch.sigmoid(self.gate_logits)

    def topk_gates(self):
        probs = self.gate_probs()

        # Non-candidate layers do not participate in Top-K
        masked_probs = probs.masked_fill(~self.candidate_mask, -1.0)

        k = min(self.target_active_layers, int(self.candidate_mask.sum().item()))
        topk_idx = torch.topk(masked_probs, k=k).indices

        hard = torch.zeros_like(probs)
        hard[topk_idx] = 1.0

        if self.training:
            # STE: forward uses hard top-k, backward follows the soft probability
            gates = hard.detach() - probs.detach() + probs
        else:
            gates = hard

        return gates

    def gate_for_layer(self, layer_id: int):
        if self.use_hard_topk:
            return self.topk_gates()[layer_id]
        else:
            return self.gate_probs()[layer_id]

    def regularization_loss(
        self,
        lambda_binary: float = 1e-3,
        lambda_margin: float = 1e-2,
        margin: float = 0.2,
    ):
        probs = self.gate_probs()
        candidate_probs = probs[self.candidate_mask]

        # 1. Push gates toward 0 or 1
        binary_loss = (candidate_probs * (1.0 - candidate_probs)).mean()

        # 2. Widen the gap between the K-th and (K+1)-th layers
        sorted_probs = torch.sort(candidate_probs, descending=True).values
        k = min(self.target_active_layers, sorted_probs.numel())

        if k < sorted_probs.numel():
            topk_min = sorted_probs[k - 1]
            rest_max = sorted_probs[k]
            margin_loss = torch.relu(margin - (topk_min - rest_max))
        else:
            margin_loss = torch.tensor(
                0.0,
                device=probs.device,
                dtype=probs.dtype
            )

        return lambda_binary * binary_loss + lambda_margin * margin_loss

    def gate_stats(self):
        with torch.no_grad():
            probs = self.gate_probs()
            hard = torch.zeros_like(probs)

            masked_probs = probs.masked_fill(~self.candidate_mask, -1.0)
            k = min(self.target_active_layers, int(self.candidate_mask.sum().item()))
            topk_idx = torch.topk(masked_probs, k=k).indices
            hard[topk_idx] = 1.0

        return probs.detach(), hard.detach()

class KgAdapterLayerInsert(nn.Module):
    def __init__(
        self,
        init_prob: float = 0.5,
        threshold: float = 0.5,
        lambda_sparse: float = 1e-4,
        use_hard_gate: bool = True,
        layer_id: int = None,
        gate_manager: nn.Module = None,
    ):
        super().__init__()

        assert 0.0 < init_prob < 1.0

        init_prob_tensor = torch.tensor(init_prob, dtype=torch.float32)
        init_logit = torch.log(init_prob_tensor / (1.0 - init_prob_tensor))

        # keep the original independent gate as a fallback
        self.gate_logit = nn.Parameter(init_logit.clone())

        self.threshold = threshold
        self.lambda_sparse = lambda_sparse
        self.use_hard_gate = use_hard_gate

        self.layer_id = layer_id
        self.gate_manager = gate_manager

    def set_manager(self, gate_manager, layer_id: int):
        self.gate_manager = gate_manager
        self.layer_id = layer_id

    def gate_prob(self):
        return torch.sigmoid(self.gate_logit)

    def binary_gate(self):
        # prefer the global Top-K gate
        if self.gate_manager is not None and self.layer_id is not None:
            return self.gate_manager.gate_for_layer(self.layer_id)

        # fallback: the original independent gate
        p = self.gate_prob()

        if self.use_hard_gate:
            z_hard = (p >= self.threshold).to(dtype=p.dtype)
            if self.training:
                z = z_hard.detach() - p.detach() + p
            else:
                z = z_hard
        else:
            z = p

        return z

    def forward(
        self,
        main_out: torch.Tensor,
        adapter_out: torch.Tensor = None,
        extra_out: torch.Tensor = None,
        return_gate: bool = False,
    ):
        if adapter_out is None:
            if return_gate:
                return main_out, None
            return main_out

        side_out = adapter_out

        if extra_out is not None:
            side_out = side_out + extra_out

        z = self.binary_gate().to(
            device=main_out.device,
            dtype=main_out.dtype
        )

        output = main_out + z * side_out

        if return_gate:
            return output, z

        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3VLConfig ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config)
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            sg=None,
            attention_mask: Optional[Dict] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Dict] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Optional[Dict]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        extra_outputs = {'input_hidden': hidden_states}

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask['text_self_mask'],
            position_ids=position_ids,
            past_key_value=past_key_value['sa_key_value'] if past_key_value is not None else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        extra_outputs['sa_hidden'] = hidden_states
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        extra_outputs['ffn_hidden'] = hidden_states
        hidden_states = residual + hidden_states

        outputs = {'text_hidden_states': hidden_states, 'extra_outputs': extra_outputs}

        if output_attentions:
            outputs['self_attn_weights'] = self_attn_weights

        if use_cache:
            outputs['sa_key_value'] = present_key_value

        return outputs
# layer 4
class Qwen3WithKgAdapterDecoderlayer(nn.Module):
    """Qwen3 decoder layer augmented with the TKG-Adapter (MoE + cross-attention).

    Reuses the base model's attention/MLP/norm weights, then injects the
    ``LowRankAdapterRouter`` and layer-insertion gate to fuse KG information.
    """

    def __init__(self, pre_model: Qwen3VLForConditionalGeneration, config: Qwen3VLConfig, num_layer: int, kg_adapter_dec_range: list):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = num_layer
        self.kg_adapter_dec_range = kg_adapter_dec_range

        # ==================== Ablation experiment config read ====================
        self.use_pure_lora = getattr(config, 'use_pure_lora', False)
        self.use_modality_image = getattr(config, 'use_modality_image', False)
        self.use_modality_graph = getattr(config, 'use_modality_graph', False)
        self.use_cross_attention = getattr(config, 'use_cross_attention', False)

        # config parameters
        self.exp_set = config.exp_set
        self.use_prefix = config.use_prefix
        self.info_merge_pos = config.info_merge_pos
        self.scaling_rate = config.scaling_rate
        self.linear_scale = config.linear_scale
        self.use_gnn = config.use_gnn
        self.output_sg = config.output_sg
        self.keep_ratio = config.keep_ratio
        self.use_trips = config.use_trips
        self.kg_adapter_cross_attention = config.kg_adapter_cross_attention
        self.kg_adapter_moe = config.kg_adapter_moe
        self.no_res = config.no_res
        self.fuse_rate = config.fuse_rate
        self.text2graph = config.text2graph
        self.graph2text = config.graph2text
        self.num_experts = config.num_experts
        self.last_token = config.last_token
        self.use_vision_adapter = True
        # self.attention_type = config.text_config.layer_types[num_layer]
        self.attention_type = "full_attention"

        ################## Text decoder components #######################
        if pre_model is not None:
            language_model = pre_model.model.language_model
            self.input_layernorm = language_model.layers[num_layer].input_layernorm
            self.self_attn = language_model.layers[num_layer].self_attn
            self.post_attention_layernorm = language_model.layers[num_layer].post_attention_layernorm
            self.mlp = language_model.layers[num_layer].mlp
            self.rotary_emb = language_model.rotary_emb
        else:
            self.self_attn = Qwen3Attention(config)
            self.mlp = Qwen3MLP(
                hidden_size=self.hidden_size,
                intermediate_size=getattr(config, 'intermediate_size', 11008),
                hidden_act=getattr(config, 'hidden_act', 'silu'),
            )
            self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.act_fn = ACT2FN[config.hidden_act]
        self.relu = nn.ReLU()

        # ============== KG Adapter components ==============
        if not self.use_pure_lora:
            # fusion of moe_out and cross_attn_out
            self.TKGAdapter_cross_merge = KgAdapterInfoMerge(
                config.kg_adapter_info_merge, config.hidden_size
            )

            self.TKGAdapter_layer_insert = KgAdapterLayerInsert(
                init_prob=getattr(config, "kg_adapter_insert_init_prob", 0.5),
                threshold=getattr(config, "kg_adapter_insert_threshold", 0.5),
                lambda_sparse=getattr(config, "kg_adapter_sparse_lambda", 1e-4),
                use_hard_gate=getattr(config, "kg_adapter_use_hard_gate", False),
            )


            self.TKGAdapter_output_info_merge = KgAdapterInfoMerge(config.kg_adapter_info_merge, config.hidden_size)

            # the Router decides internally what to build based on config
            self.LowRankAdapterRouter = LowRankAdapterRouter(
                KgAdapterCrossAttention,
                Qwen3RMSNorm,
                KgAdapterMLP,
                config,
                config.num_experts,
                config.hidden_size,
                config.image_hidden_size,
                config.graph_hidden_size,
                config.topk,
                config.kg_adapter_hidden_size,
            )

    def forward(
        self,
        text_hidden_states: torch.Tensor,
        attention_mask: Optional[Dict] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Dict] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        image_position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        pre_graph_hidden_states: Optional[torch.Tensor] = None,
        graph_hidden_states: Optional[torch.Tensor] = None,
        sg=None,
        attention_mask_dict: Optional[Dict] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        batch_size = text_hidden_states.shape[0]

        # ==================== 1. Text backbone processing ====================
        text_residual = text_hidden_states
        text_hidden_states = self.input_layernorm(text_hidden_states)

        text_hidden_states, self_attn_weights = self.self_attn(
            hidden_states=text_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        text_hidden_states = text_residual + text_hidden_states
        text_residual = text_hidden_states
        text_hidden_states = self.post_attention_layernorm(text_hidden_states)

        # ==================== 2. LoRA baseline interception ====================
        if getattr(self, 'use_pure_lora', False):
            text_hidden_states = self.mlp(text_hidden_states)
            text_hidden_states = text_residual + text_hidden_states
            outputs = {'text_hidden_states': text_hidden_states, 'graph_hidden_states': None}
            if use_cache:
                outputs['sa_key_value'] = past_key_values
            return outputs
        
        if self.layer_id not in self.kg_adapter_dec_range:
            text_hidden_states = self.mlp(text_hidden_states)
            text_hidden_states = text_residual + text_hidden_states
            
            # Important detail: pass graph_hidden_states through instead of None
            outputs = {
                'text_hidden_states': text_hidden_states, 
                'graph_hidden_states': graph_hidden_states # pass through to the next layer!
            }
            if use_cache:
                outputs['sa_key_value'] = past_key_values
            return outputs

        # ==================== 3. Modality processing ====================
        if getattr(self, 'use_modality_graph', True) and graph_hidden_states is not None and pre_graph_hidden_states is not None:
            graph_hidden_states = pre_graph_hidden_states + graph_hidden_states
        else:
            graph_hidden_states = None

        # ==================== 4. Split image and text in the main sequence ====================
        use_image_split = (
            getattr(self, 'use_modality_image', False)
            and image_mask is not None
            and image_mask.any()
        )

        if use_image_split:
            mask_bool = image_mask.bool()  # [B, S, H] - exactly the same shape as text_hidden_states

            image_only_states = torch.zeros_like(text_hidden_states)
            text_only_states  = torch.zeros_like(text_hidden_states)

            image_only_states = torch.where(mask_bool, text_hidden_states, image_only_states)
            text_only_states  = torch.where(~mask_bool, text_hidden_states, text_only_states)
        else:
            image_only_states = None
            text_only_states  = text_hidden_states

        # ==================== 5. Routing ====================
        results = self.LowRankAdapterRouter(
            attention_mask=attention_mask_dict,
            position_ids=position_ids,
            text_states=text_only_states,
            image_states=image_only_states,
            hidden_states=text_hidden_states,
            graph_states=graph_hidden_states,
        )

        text_router_out  = results.get('text',    None)
        graph_router_out = results.get('graph',      None)
        image_router_out = results.get('image',   None)
        cross_attn_out   = results.get('cross',   None)
        channel_attn_out = results.get('channel', None)

        # ==================== 6. Feature fusion ====================
        # Step 1: moe_out = text_router_out + image_router_out
        if text_router_out is not None and image_router_out is not None:
            moe_out = text_router_out + image_router_out
        elif text_router_out is not None:
            moe_out = text_router_out
        elif image_router_out is not None:
            moe_out = image_router_out
        else:
            moe_out = None

        # cross fusion: Gate(moe_out, cross_attn_out)
        cross_contribution = None
        if cross_attn_out is not None and moe_out is not None:
            moe_out = self.TKGAdapter_cross_merge(moe_out, cross_attn_out)

        
        # MLP backbone
        text_hidden_states = self.mlp(text_hidden_states)

        if moe_out is not None:
            # without cross-modal info, the Gate degenerates to fusing moe_out directly
            # text_hidden_states = self.TKGAdapter_output_info_merge(text_hidden_states, moe_out)
            text_hidden_states = self.TKGAdapter_layer_insert(main_out=text_hidden_states, adapter_out=moe_out)


        # Step 6: residual connection
        text_hidden_states = text_residual + text_hidden_states

        # graph feature pass-through
        new_graph_hidden_states = graph_router_out if graph_router_out is not None else graph_hidden_states

        outputs = {
            'text_hidden_states':  text_hidden_states,
            'graph_hidden_states': new_graph_hidden_states,
        }
        if use_cache:
            outputs['sa_key_value'] = past_key_values

        return outputs


class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3VLConfig 
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        b = 1       


    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Qwen3VLForConditionalGeneration):
            module.gradient_checkpointing = value


def update_flat_sg(sg, exp_set="nodes"):
    # x, edge -> trip_rep
    if "use_cat_trips" in exp_set:
        trip_ids = sg.trips['trip_ids']
        node_mask = sg.trips['node_mask']
        edge_mask = sg.trips['edge_mask']
        trip_rep = sg.trip_rep.to(sg.x.dtype)

        trip_rep[node_mask] = sg.x
        trip_rep[edge_mask] = sg.edge_rep.to(sg.x.dtype)

        return trip_rep

    elif "use_trips" in exp_set:
        trip_num = sg.trips['trip_num']
        trip_rep = sg.trip_rep
        # <h>S<r>R<t>O<h>S<r>R<t>O....
        for bs in range(trip_rep.size(0)):
            max_trip_num = (trip_num[bs + 1] - trip_num[bs]) * 3
            trip_rep[bs, [x for x in range(0, max_trip_num, 3)]] = sg.x.index_select(0, sg.edge_index[0][
                                                                                        trip_num[bs]: trip_num[
                                                                                            bs + 1]])
            trip_rep[bs, [x for x in range(1, max_trip_num, 3)]] = sg.edge_rep[trip_num[bs]: trip_num[bs + 1]]
            trip_rep[bs, [x for x in range(2, max_trip_num, 3)]] = sg.x.index_select(0, sg.edge_index[1][
                                                                                        trip_num[bs]: trip_num[
                                                                                            bs + 1]])
        return trip_rep

    # x -> node_rep
    else:

        node_rep = sg.node_rep.clone()  # clone to keep the original values
        node_mask = sg.node_mask

        # make sure node_mask is a 1D boolean tensor
        mask_flat = node_mask.bool().view(-1)

        # only update the values at masked positions
        node_rep.view(-1, sg.x.size(-1))[mask_flat] = sg.x.view(-1, sg.x.size(-1))[mask_flat]

        return node_rep


def update_structure_sg(sg, exp_set="nodes"):
    # trip_rep -> x, edge_rep
    if "use_cat_trips" in exp_set:
        trip_ids = sg.trips['trip_ids']
        node_mask = sg.trips['node_mask']
        edge_mask = sg.trips['edge_mask']
        trip_rep = sg.trip_rep
        sg.x = trip_rep[node_mask]
        sg.edge_rep = trip_rep[edge_mask]
        return sg
    # trip_rep -> x, edge
    elif "use_trips" in exp_set:
        trip_rep = sg.trip_rep.view(-1, sg.trip_rep.size(-1))
        trip_ids = sg.trips['trip_ids'].view(-1)

        sg.edge_rep = trip_rep[trip_ids < 0]

        node_idx = [-1 for x in range(sg.x.size(0))]
        used = set()
        tmp = (trip_ids[trip_ids > 0] - 1).tolist()
        for tid, nid in enumerate(tmp):
            if nid not in used:
                node_idx[nid] = tid
                used.add(nid)
        sg.x = trip_rep[trip_ids > 0].index_select(0, torch.tensor(node_idx, device=trip_rep.device))

        return sg
    # node_rep -> x
    else:
        node_rep = sg.node_rep
        node_mask = sg.node_mask
        # sg.x = node_rep[node_mask.bool()]
        sg.x = torch.zeros_like(node_rep).view(-1, node_rep.size(-1))
        sg.x[node_mask.bool().view(-1)] = node_rep.view(-1, node_rep.size(-1))[node_mask.bool().view(-1)]

        return sg


def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    from torch.autograd import Variable
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target


def _build_new_ca_attention_mask(attention_mask, input_shape, inputs_embeds):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )
    return combined_attention_mask

#Qwen3VLModel.language_model, the main model
#layer 3
class Qwen3WithKgAdapterModel(Qwen3PreTrainedModel):
    def __init__(self, pre_model, config: TKGAdapterConfig):
        _tc = getattr(config, 'text_config', None)
        if _tc is not None:
            for _attr in ['initializer_range', 'rms_norm_eps', 'hidden_act',
              'hidden_size', 'num_hidden_layers', 'intermediate_size']:
                if hasattr(_tc, _attr):
                    setattr(config, _attr, getattr(_tc, _attr))
        config.attn_implementation = "eager"
        config._attn_implementation = "eager"
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.use_node_emb = config.use_node_emb
        self.use_edge_emb = config.use_edge_emb
        self.mix_emb = config.mix_emb
        self.use_trips = config.use_trips
        self.output_sg = config.output_sg
        self.config = config
        self.linear_emb = config.linear_emb
        self.gradient_checkpointing = True
        # self.gradient_checkpointing = False
        # embedding layer

        if self.use_node_emb:
            self.embed_nodes = pre_model.embed_nodes
            self.kg_adapter_nodes_proj = pre_model.kg_adapter_nodes_proj
        
        if self.use_edge_emb:
            self.kg_adapter_embed_edges = pre_model.kg_adapter_embed_edges
            self.kg_adapter_edges_proj = pre_model.kg_adapter_edges_proj
        
        self.adapter_router_manager = pre_model.adapter_router_manager
        self.layers = pre_model.layers


        candidate_layer_ids = []

        for layer in self.layers:
            if hasattr(layer, "layer_id") and hasattr(layer, "kg_adapter_dec_range"):
                if layer.layer_id in layer.kg_adapter_dec_range:
                    candidate_layer_ids.append(layer.layer_id)

        self.layer_insert_manager = KgAdapterLayerInsertManager(
            num_layers=len(self.layers),
            target_active_layers=getattr(config, "target_active_layers", 8),
            candidate_layer_ids=candidate_layer_ids,
            init_prob=getattr(config, "kg_adapter_insert_init_prob", 0.5),
            init_noise_std=getattr(config, "kg_adapter_insert_init_noise_std", 0.2),
            use_hard_topk=getattr(config, "kg_adapter_use_hard_topk", True),
        )

        for layer in self.layers:
            if hasattr(layer, "TKGAdapter_layer_insert"):
                layer.TKGAdapter_layer_insert.set_manager(
                    self.layer_insert_manager,
                    layer.layer_id
                )

        # self.language_model = pre_model.model.language_model
        self.embed_tokens = pre_model.model.language_model.embed_tokens
        self.rotary_emb = pre_model.model.language_model.rotary_emb
        self.norm = pre_model.model.language_model.norm
        # self.has_sliding_layers = "sliding_attention" in self.config.text_config.layer_types

        self.has_sliding_layers = False
        self.act_fn = ACT2FN[config.hidden_act]
        
        self.init_weights()

    def init_weights(self):
        std = self.config.initializer_range
        print('Start initializing custom adapter weights...')

        # ================= 1. Keep this part (Embedding outside the layers) =================
        # initialize the node embedding layer
        if self.use_node_emb:
            nn.init.normal_(self.embed_nodes.weight, 0, std)
            if self.embed_nodes.padding_idx is not None:
                self.embed_nodes.weight.data[self.embed_nodes.padding_idx].zero_()
            # self.kg_adapter_nodes_proj
            nn.init.normal_(self.kg_adapter_nodes_proj.weight, 0, std)
            if self.kg_adapter_nodes_proj.bias is not None:
                nn.init.zeros_(self.kg_adapter_nodes_proj.bias)

        # initialize the edge embedding layer
        if self.use_edge_emb:
            for layer in self.kg_adapter_embed_edges:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, 0, std)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            # self.kg_adapter_edges_proj
            nn.init.normal_(self.kg_adapter_edges_proj.weight, 0, std)
            if self.kg_adapter_edges_proj.bias is not None:
                nn.init.zeros_(self.kg_adapter_edges_proj.bias)

        if self.mix_emb:
            # initialize the mixing factor
            self.kg_adapter_t2n_mix_facotr.data.zero_()
            self.kg_adapter_t2e_mix_facotr.data.zero_()
            # self.kg_adapter_t2n_proj
            nn.init.normal_(self.kg_adapter_t2n_proj.weight, 0, std)
            if self.kg_adapter_t2n_proj.bias is not None:
                nn.init.zeros_(self.kg_adapter_t2n_proj.bias)
            # self.kg_adapter_t2e_proj
            nn.init.normal_(self.kg_adapter_t2e_proj.weight, 0, std)
            if self.kg_adapter_t2e_proj.bias is not None:
                nn.init.zeros_(self.kg_adapter_t2e_proj.bias)

        # ================= 2. Modify this part (Adapter inside the layers) =================
        # define the signature words of new modules
        NEW_MODULE_KEYWORDS = [
            "kg_adapter", 
            "TKGAdapter", 
            "LowRankAdapterRouter",
            "TKGAdapter_moe",
            "KgAdapterCrossAttention", 
            "KgAdapterMLP",
        ]

        for i, module in enumerate(self.layers):
            for name, child in module.named_modules():
                # 1. Check whether it is a new module (name contains adapter/router etc.)
                is_new_module = any(k in name for k in NEW_MODULE_KEYWORDS)
                
                # 2. Explicitly protect the vision layers (prevent vision params from being reset)
                is_visual = "visual" in name

                # Initialize only when it is a new module and not a vision layer
                if is_new_module and not is_visual:
                    if isinstance(child, nn.Linear):
                        # print(f"Initializing Layer {i}: {name}")
                        nn.init.normal_(child.weight, 0, std)
                        if child.bias is not None:
                            nn.init.zeros_(child.bias)
                    elif isinstance(child, nn.Embedding):
                        nn.init.normal_(child.weight, 0, std)
                        if child.padding_idx is not None:
                            child.weight.data[child.padding_idx].zero_()

        print('Initialization parameters completed!')

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length=None,
                                        type="sa"):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1 and type == "sa":
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    def get_text_hidden_states(self, attention_mask, input_shape, inputs_embeds, past_key_values_length=None,
                                        type="sa"):
        
        return None
    
    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ):
        """Add intermediate vision-encoder features to the language-model hidden states at the matching positions."""
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states
    
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            # add
            pre_graph_hidden_states: torch.Tensor = None,
            graph_emb: torch.Tensor = None,
            image_mask: Optional[torch.Tensor] = None,  
            visual_pos_masks: Optional[torch.Tensor] = None,
            deepstack_visual_embeds: Optional[list] = None,
            sg = None,
            attention_mask_dict: Optional[Dict] = None, 
     ) -> Union[Tuple, BaseModelOutputWithPast]:
         # ==================== Original processing ====================
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_sg_states = True if self.output_sg else None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            # inputs_embeds = self.language_model.embed_tokens(input_ids)
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
            text_position_ids = None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        # ==================== Global mask initialization ====================
        # these base masks must exist regardless of whether the graph modality is on
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=device
            )
            
        text_self_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, type="sa"
        )   
        
        # pre-initialize all graph-related masks to None so dict assembly never raises UnboundLocalError
        node_text_mask = None
        text_node_mask = None
        align_mask = None
        trip_text_mask = None
        text_trip_mask = None
        graph_hidden_states = None

        # ==================== Graph processing logic (controlled execution) ====================
        use_modality_graph = getattr(self.config, 'use_modality_graph', True)
        
        if use_modality_graph and graph_emb is not None:
            # whether to use node embeddings
            nodes_embeds = None
            if self.use_node_emb:
                if self.linear_emb:
                    nodes_embeds = self.embed_nodes(sg.node_ids).view(batch_size, -1, self.config.kg_adapter_node_emb_size)
                else:
                    nodes_embeds = self.act_fn(self.embed_nodes(sg.node_ids).view(batch_size, -1, self.config.kg_adapter_node_emb_size))
                
            # whether to use the LLM encoder to encode node embeddings, then mix with the original encoding
            if self.mix_emb:    # this is subword fusion after the entity passes through the LLM
                if nodes_embeds is None:
                    nodes_embeds = self.act_fn(self.kg_adapter_t2n_proj(self.embed_tokens(sg.nid2swid).sum(2)))
                else:
                    if self.linear_emb:
                        nodes_embeds = self.act_fn(nodes_embeds + self.kg_adapter_t2n_proj(self.embed_tokens(sg.nid2swid).sum(2)))
                    else:
                        t2n_gate = torch.sigmoid(self.kg_adapter_t2n_mix_facotr)
                        nodes_embeds = self.act_fn(t2n_gate * nodes_embeds + (1 - t2n_gate) * self.kg_adapter_t2n_proj(
                            self.embed_tokens(sg.nid2swid).sum(2)))

            # embed edges
            edges_embeds = None
            if self.use_edge_emb:
                edge_vec = make_one_hot(sg.edge_type, self.config.num_relations * 2 + 1)
                node_type = sg.n_id.view(-1).contiguous()  # [`total_n_nodes`, ]
                head_type = node_type[sg.edge_index[0]]
                tail_type = node_type[sg.edge_index[1]]  # [E,] #tail=tgt
                head_vec = make_one_hot(head_type, len(sg.node_ids[0]))  # [E,5]
                tail_vec = make_one_hot(tail_type, len(sg.node_ids[0]))  # [E,5]
                headtail_vec = torch.cat([head_vec, tail_vec], dim=1)  # [E,10]
                edge_feature = torch.cat([edge_vec, headtail_vec], dim=1).to(torch.float32)
                edges_embeds = self.kg_adapter_embed_edges(edge_feature)  # [E, emb_dim]
            
            if self.use_trips:
                bsz = len(sg.ptr) - 1
                max_trip_num = max(sg.num_edges).item()
                edge_ids = unbatch_edge_index(sg.edge_index, sg.batch)
                # node_rep, edge_rep -> trip_rep
                batch_trip_mask = []
                for bs in range(bsz):
                    edge_idx = edge_ids[bs]
                    batch_trip_mask.append(
                        torch.cat([torch.ones(edge_idx.size(1)), torch.zeros((max_trip_num - edge_idx.size(1)))], dim=0))

                sg.trip_mask = torch.stack(batch_trip_mask).to(sg.x.device)
                sg.trip_rep = torch.zeros((sg.trip_mask.size(0), sg.trip_mask.size(1), sg.x.size(1))).to(sg.x.device)
            
            # set the graph feature variables passed downstream
            pre_graph_hidden_states = graph_emb  # pretrained graph embedding
            graph_hidden_states = graph_emb # current graph embedding
        
            # ==================== Graph-related attention mask construction ====================
            # graph_mask
            graph_mask = torch.ones(
                (graph_emb.size(0), graph_emb.size(1)),
                dtype=torch.long,
                device=graph_emb.device  # add this line
            )
            text_node_mask = self._prepare_decoder_attention_mask(
                graph_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, type="ca"
            )
            
            if self.config.align_mask:
                if past_key_values_length != 0:
                    align_mask = torch.cat([torch.zeros(batch_size, seq_length_with_past - sg.align_mask.size(1),
                                                        sg.align_mask.size(-1), device=inputs_embeds.device),
                                            sg.align_mask], dim=1)
                else:
                    align_mask = sg.align_mask
            if self.use_trips:
                trip_text_mask = self._prepare_decoder_attention_mask(
                    attention_mask, (sg.trip_rep.size(0), sg.trip_rep.size(1)), inputs_embeds, past_key_values_length,
                    type="ca"
                )
                text_trip_mask = self._prepare_decoder_attention_mask(
                    sg.trip_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, type="ca"
                )

        # unified Attention Mask dict
        # the dict structure stays consistent whether the graph is on or off; only some values are None,
        # so downstream code never crashes
        attention_mask_dict = {
            "base_mask": attention_mask, # padding mask
            "text_self_mask": text_self_mask, # self-attention mask
            "node_text_mask": node_text_mask, # node-text mask (cross attention)
            "text_node_mask": text_node_mask, # text-node mask (cross attention)
            "align_mask": align_mask,
            "trip_text_mask": trip_text_mask,
            "text_trip_mask": text_trip_mask,
            "seq_length": seq_length,
        }

        hidden_states = inputs_embeds
        # position_embeddings = self.language_model.rotary_emb(hidden_states, position_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_sg_states = () if output_sg_states else None
        next_decoder_cache = () if use_cache else None


        def create_custom_forward(module):
            def custom_forward(
                hidden_states,
                attention_mask,
                position_ids,
                cache_position,
                position_embeddings,
                image_mask,
                graph_hidden_states,
                pre_graph_hidden_states,
            ):
                return module(
                    text_hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    attention_mask_dict=attention_mask_dict,
                    graph_hidden_states=graph_hidden_states,
                    pre_graph_hidden_states=pre_graph_hidden_states,
                    sg=sg,
                    image_mask=image_mask,
                )
            return custom_forward

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    causal_mask_mapping[decoder_layer.attention_type],
                    text_position_ids,
                    cache_position,
                    position_embeddings,
                    image_mask,
                    graph_hidden_states,
                    pre_graph_hidden_states,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    text_hidden_states=hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    # modality params (graph-related params are safely None even when the graph modality is off)
                    attention_mask_dict=attention_mask_dict,
                    graph_hidden_states=graph_hidden_states,
                    pre_graph_hidden_states=pre_graph_hidden_states,
                    sg=sg,
                    image_mask=image_mask,
                )

            hidden_states = layer_outputs['text_hidden_states']
            if (deepstack_visual_embeds is not None
                    and visual_pos_masks is not None
                    and idx < len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[idx],
                )
            graph_hidden_states = layer_outputs.get('graph_hidden_states', None)

            if output_attentions:
                attn_weights = layer_outputs.get('self_attn_weights', None) 
                if attn_weights is None and len(layer_outputs) > 1:
                     attn_weights = layer_outputs[1] 
                all_self_attns += (attn_weights,)

            if use_cache:
                cache = {'sa_key_value': layer_outputs['sa_key_value']}
                if 'ca_n2t_key_value' in layer_outputs:
                    cache['ca_n2t_key_value'] = layer_outputs['ca_n2t_key_value']
                if 'ca_t2n_key_value' in layer_outputs:
                    cache['ca_t2n_key_value'] = layer_outputs['ca_t2n_key_value']
                next_decoder_cache += (cache,)

            if output_sg_states and "sg_state" in layer_outputs:
                all_sg_states += (layer_outputs['sg_state'],)

        # hidden_states = self.language_model.norm(hidden_states)
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_sg_states,  
        )
    
class QwenKgAdapterForCausalLM(Qwen3VLPreTrainedModel):
    """CausalLM wrapper that replaces the decoder with the KG-adapter decoder."""

    def __init__(self, pre_model, config):
        super().__init__(config)

        # build the TKGAdapter model
        config.attn_implementation = "eager" 
        self.visual = pre_model.model.visual
        self.language_model = Qwen3WithKgAdapterModel(pre_model, config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        #self.post_init()  # initializes embed_tokens and Linear layers in components


    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        # videos are expanded frame by frame; each frame is processed independently
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1],
                dtype=input_ids.dtype, device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # t_index is always 0; video temporal info is encoded via timestamp tokens
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.
        The deepstack visual features are also returned.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    @staticmethod
    def is_torchdynamo_compiling():
        return torch.compiler.is_compiling() if hasattr(torch, 'compiler') else False

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            # add
            graph_emb: torch.Tensor = None,
            labels: Optional[torch.LongTensor] = None,
            sg = None,
    ) -> Union[Tuple, Qwen3VLCausalLMOutputWithPast]:
        """
        Forward pass for QwenKgAdapterForCausalLM.
        
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Position indices for input tokens.
            past_key_values (`List[torch.FloatTensor]`, *optional*):
                Cached past key values for faster decoding.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Input embeddings instead of input_ids.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss.
            use_cache (`bool`, *optional*):
                Whether to use the cache for faster generation.
            output_attentions (`bool`, *optional*):
                Whether to output attentions weights.
            output_hidden_states (`bool`, *optional*):
                Whether to output hidden states.
            return_dict (`bool`, *optional*):
                Whether to return a ModelOutput instead of a plain tuple.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
                The rope index difference between sequence length and multimodal rope.
        Returns:
            model output
        """
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds = None
        image_mask = None
        visual_pos_masks = None
        deepstack_visual_embeds = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # assemble visual_pos_masks and deepstack_visual_embeds
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None:
            visual_pos_masks = image_mask[..., 0]          # [B, S]
            deepstack_visual_embeds = deepstack_image_embeds  # list of 3 tensors
        
        # In QwenKgAdapterForCausalLM.forward, after inputs_embeds is processed and
        # before language_model is called, insert the following:

        if position_ids is None:
            attention_mask_for_rope = attention_mask
            # if attention_mask is 4D, restore it to 2D
            if attention_mask_for_rope is not None and attention_mask_for_rope.ndim == 4:
                attention_mask_for_rope = torch.diagonal(
                    attention_mask_for_rope[:, 0], dim1=1, dim2=2
                )
                if attention_mask_for_rope.dtype.is_floating_point:
                    attention_mask_for_rope = (
                        1.0 - attention_mask_for_rope / torch.finfo(attention_mask_for_rope.dtype).min
                    ).int()

            prefill_noncompiled_stage = (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if prefill_noncompiled_stage or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_for_rope,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        #Qwen3VLModel.language_model's output
        outputs = self.language_model( # Qwen3WithKgAdapterModel's forward runs here
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            attention_mask_dict={"base_mask": attention_mask} if attention_mask is not None else None,
            # add
            pre_graph_hidden_states=graph_emb,
            graph_emb=graph_emb,
            sg=sg,
            image_mask=image_mask,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        # output of Qwen3VLModel
        output = Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()

#layer 1
class QwenVLForConditionalGeneration(Qwen3VLPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    def __init__(self, pre_model, config):
        super().__init__(config)
        self.model = QwenKgAdapterForCausalLM(pre_model, config)
        #self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.lm_head = pre_model.lm_head
        # self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.
        The deepstack visual features are also returned.
        """
        # directly reuse get_image_features, consistent with the official implementation
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    # @can_return_tuple
    # @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # add
        graph_emb: torch.Tensor = None,
        sg = None,
        **kwargs
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.


        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        >>> model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            # add
            graph_emb=graph_emb,
            labels=labels,
            sg = sg,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(hidden_states[:, slice_indices, :])

        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            # logits_to_keep=0 means keep all (training mode)
            logits = self.lm_head(hidden_states)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size
            )

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        # keep your custom params
        graph_emb=None,
        sg=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen3-VL position_ids are computed uniformly inside forward; set to None here
        model_inputs["position_ids"] = None

        # images need not be passed again during the non-prefill stage
        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        # pass through custom params
        model_inputs["graph_emb"] = graph_emb
        model_inputs["sg"] = sg

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

def get_model(model, model_config):
    """Build the decoder layer stack with KG-Adapter layers.

    Args:
        model: The base Qwen3-VL model.
        model_config: Config carrying adapter/modality flags and layer ranges.

    Returns:
        The base model with its ``language_model.layers`` replaced by
        ``Qwen3WithKgAdapterDecoderlayer`` instances.
    """
    # ==================== Base model config ====================
    # set the model padding index (for pad tokens), 2
    model.padding_idx = model_config.pad_token_id
    # set the vocab size, 32000
    model.vocab_size = model_config.vocab_size
    # set the hidden dim, 2560
    model.hidden_size = model_config.hidden_size
    
    # ==================== TKGAdapter feature-switch config ====================
    # whether to use node embeddings (entity representations learned from the KG)
    model.use_node_emb = model_config.use_node_emb
    # whether to use edge embeddings (relation representations learned from the KG)
    model.use_edge_emb = model_config.use_edge_emb
    # whether to use triple info (full head-relation-tail structure)
    model.use_trips = model_config.use_trips
    # whether to output the subgraph structure (for visualization/further processing)
    model.output_sg = model_config.output_sg
    # whether to apply a linear transform to the embeddings
    model.linear_emb = model_config.linear_emb
    # set the activation type (e.g. ReLU, GELU)
    model.act_fn = ACT2FN[model_config.hidden_act]

    # ==================== TKGAdapter layer-range config ====================
    # compute the decoder-layer step for flexible layer selection
    # use the configured step if given, else 1 (contiguous layers); here [0, 32] are all inserted
    dec_stride = model_config.kg_adapter_dec_range[2] if len(model_config.kg_adapter_dec_range) == 3 else 1
    # generate the list of layer indices where TKGAdapter is actually inserted
    # e.g. range(0, 16, 1) inserts TKGAdapter into the first 16 layers
    model.kg_adapter_dec_range = [x for x in
                                range(model_config.kg_adapter_dec_range[0], 
                                      model_config.kg_adapter_dec_range[1], 
                                      dec_stride)]

    # ==================== Model layer construction ====================
    # module list storing all layers
    module_lst = []
    # list of all routers (for MoE routing management)
    token_router_list = []
    

    # iterate over all hidden layers
    for i in range(model_config.num_hidden_layers):
        # if the current layer is within the TKGAdapter insertion range
        # if i in model.kg_adapter_dec_range:
        # create a decoder layer with TKGAdapter
        # LlamaWithKgAdapterDecLayer integrates the KG adapter
        module_lst.append(Qwen3WithKgAdapterDecoderlayer(model, model_config, i, model.kg_adapter_dec_range))
        
        # extract three routing components from the current layer's router:
        # router_text: text info routing
        # router_graph: graph info routing
        # router_cross: cross info routing
        if model_config.use_text_moe:
            token_router_list.append(module_lst[i].LowRankAdapterRouter.router_text)
        if model_config.use_graph_moe:
            token_router_list.append(module_lst[i].LowRankAdapterRouter.router_graph)
        if model_config.use_image_moe:
            token_router_list.append(module_lst[i].LowRankAdapterRouter.router_image)
        if model_config.use_cross_attention:
            token_router_list.append(module_lst[i].LowRankAdapterRouter.router_cross)
        if model_config.use_channel_attention:
            token_router_list.append(module_lst[i].LowRankAdapterRouter.router_channel)
        # else:
            # create a standard LLaMA decoder layer (without TKGAdapter)
            # module_lst.append(Qwen3DecoderLayer(model_config))
    
    # convert the layer list to a PyTorch module list so params are trainable
    model.layers = nn.ModuleList(module_lst)
    
    # ==================== Router management ====================
    # create a router manager to unify routing decisions across all layers
    # RouterManager coordinates load balancing and routing across experts
    adapter_router_manager = RouterManager(model_config, token_router_list)
    model.adapter_router_manager = adapter_router_manager

    return model

class TKGAdapterModel:
    """Factory for building the full TKG-Adapter model from a pretrained base model."""

    @staticmethod
    def from_pretrained(
            model: PreTrainedModel,
            name_or_path: Optional[str] = None,
    ) -> PeftModel:
        with open(name_or_path + "config.json") as f:
            config = json.load(f)
        if isinstance(config, dict):
            config = TKGAdapterConfig(**config)
        else:
            config = TKGAdapterConfig.from_config(config)

        # immediately replace the parent class's auto-generated wrong text_config
        _real_text_config   = model.config.text_config
        _real_vision_config = model.config.vision_config
        config.text_config   = _real_text_config
        config.vision_config = _real_vision_config
        config.hidden_size   = _real_text_config.hidden_size        # 2560
        config.image_hidden_size = _real_vision_config.hidden_size  # 1024

        model = get_model(model, config)
        model = get_tkg_adapter_model(model, config)
        model.config = config

        return model

def get_tkg_adapter_model(model: PreTrainedModel, config: PretrainedConfig) -> PeftModel:
    """Wrap the base model with the TKG-Adapter using real text/vision dims.

    Args:
        model: The base Qwen3-VL model.
        config: TKG-Adapter configuration.

    Returns:
        The adapter-augmented model.
    """
    _tc = model.config.text_config
    _vc = model.config.vision_config
    config.hidden_size       = _tc.hidden_size        # 2560
    config.image_hidden_size = _vc.hidden_size        # 1024
    config.model_type        = model.config.model_type
    model = _apply_tkg_adapter(model, config=config)
    return model


def _apply_tkg_adapter(model: PreTrainedModel, config: PretrainedConfig) -> PeftModel:
    """Instantiate the adapter model and apply the two-stage parameter freezing.

    Args:
        model: The base model.
        config: TKG-Adapter configuration.

    Returns:
        The adapter-augmented model with only adapter/router/LoRA params trainable.
    """
    # ==================== KG embedding config ====================
    if config.node_emb_path is not None:
        nodes_emb = torch.load(config.node_emb_path)
        if isinstance(nodes_emb, dict):
            nodes_emb = nodes_emb['nodes_emb']
        config.node_num = nodes_emb.size(0)
        config.kg_adapter_node_emb_size = nodes_emb.size(-1)
        print("kg nodes num: ", nodes_emb.size(0))
        del nodes_emb
    else:
        print("not use pretrained kg embedding")

    # ==================== Model initialization ====================
    if config.model_type == 'qwen3_vl':
        MODEL_CLASS = QwenVLForConditionalGeneration

    model = MODEL_CLASS(pre_model=model, config=config)

    # ==================== Round 1: base freezing strategy ====================
    # keep only adapter / lora / router / TKGAdapter params trainable
    print("freezing weights....")

    for name, param in model.named_parameters():
        # freeze all non-adapter params by default
        if ('kg_adapter' not in name and
            'embed_edges' not in name and
            'embed_nodes' not in name):
            param.requires_grad = False

        # LoRA params trainable
        if 'lora' in name:
            param.requires_grad = True

        # Router params trainable
        if 'LowRankAdapterRouter' in name:
            param.requires_grad = True

        # TKGAdapter params trainable
        if 'TKGAdapter' in name:
            param.requires_grad = True

        # KG embedding init
        if "init_kg_emb" in config.exp_set and 'embed_nodes' in name:
            from torch.nn.init import kaiming_normal_
            param.data = kaiming_normal_(param)

    # ==================== Round 2: freeze stray params by activated modality ====================
    print("正在按激活模态冻结游离参数...")

    use_modality_graph    = getattr(config, 'use_modality_graph',    False)
    use_modality_image    = getattr(config, 'use_modality_image',    False)
    use_text_moe          = getattr(config, 'use_text_moe',          True)
    use_graph_moe         = getattr(config, 'use_graph_moe',         False)
    use_image_moe         = getattr(config, 'use_image_moe',         False)
    use_cross_attention   = getattr(config, 'use_cross_attention',   False)
    use_channel_attention = getattr(config, 'use_channel_attention', False)
    use_pure_lora         = getattr(config, 'use_pure_lora',         False)

    def _should_freeze(name: str) -> bool:
        # -- Pure LoRA mode: freeze all Routers / Adapters --
        if use_pure_lora:
            # match module names exactly with dots to avoid wrongly matching e.g. .graph_to_text_proj.
            if any(k in name for k in ['.LowRankAdapterRouter.', '.TKGAdapter.']):
                return True

        # -- Text MoE inactive: freeze text router and text adapters --
        if not use_text_moe:
            if any(k in name for k in ['.router_text.', '.text_adapters.']):
                return True
        else:
            if '.text_proj.' in name and '.router_text.' not in name and 'graph_to_text_proj' not in name:
                return True

        # -- Graph modality / graph MoE inactive --
        if not use_modality_graph or not use_graph_moe:
            if any(k in name for k in ['.router_graph.', '.graph_adapters.']):
                return True
        if use_modality_graph and not use_graph_moe:
            # when graph_moe is off, graph_proj is used; freeze graph_adapters/router_graph
            if any(k in name for k in ['.router_graph.', '.graph_adapters.']):
                return True
        if not use_modality_graph:
            # graph modality fully off: graph_proj does not participate in forward
            if '.graph_proj.' in name:
                return True

        # -- Image modality / image MoE inactive --
        if not use_modality_image or not use_image_moe:
            if any(k in name for k in ['.router_image.', '.image_adapters.']):
                return True
        if not use_modality_image:
            if '.image_proj.' in name:
                return True

        # -- Cross-modal attention inactive --
        if not use_cross_attention:
            if any(k in name for k in [
                '.router_cross.', '.cross_adapters.',
                '.text_downscale.', '.graph_downscale.', '.cross_upscale.'
            ]):
                return True

        # -- Channel attention inactive --
        if not use_channel_attention:
            if any(k in name for k in ['.router_channel.', '.channel_adapters.']):
                return True

        return False

    _frozen_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and _should_freeze(name):
            param.requires_grad = False
            _frozen_count += 1
            print(f"  [冻结游离参数] {name}")

    _trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"冻结完成：冻结 {_frozen_count} 个游离参数，剩余 {_trainable} 个可训练参数")

    return model