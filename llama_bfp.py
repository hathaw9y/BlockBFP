import types
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from llama_rotation import apply_hadamard_to_last_dim

try:
    from transformers.models.llama import modeling_llama

    apply_rotary_pos_emb = modeling_llama.apply_rotary_pos_emb
    repeat_kv = modeling_llama.repeat_kv
except ImportError:
    apply_rotary_pos_emb = None
    repeat_kv = None


BFP_DEFAULT_BLOCK_SIZE = 32


def resolve_bfp_block_size(groupsize):
    return BFP_DEFAULT_BLOCK_SIZE if groupsize == -1 else groupsize


def clamp_to_dtype_range(x, dtype):
    if not torch.is_floating_point(x):
        return x
    finfo = torch.finfo(dtype)
    return torch.clamp(x, min=finfo.min, max=finfo.max)


def bfp_fake_quant(x, bits=4, block_size=BFP_DEFAULT_BLOCK_SIZE, clip_ratio=1.0):
    if bits < 2:
        raise ValueError(f"BFP bits must be at least 2, got {bits}.")
    if block_size <= 0:
        raise ValueError(f"BFP block size must be positive, got {block_size}.")

    minq = -(2 ** (bits - 1))
    maxq = 2 ** (bits - 1) - 1
    orig_shape = x.shape
    orig_dtype = x.dtype
    orig_finfo = torch.finfo(orig_dtype)

    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x = x.to(dtype=compute_dtype)
    finfo = torch.finfo(compute_dtype)
    x = torch.nan_to_num(x, nan=0.0, posinf=finfo.max, neginf=-finfo.max)

    pad = (block_size - x.shape[-1] % block_size) % block_size
    if pad:
        x = F.pad(x, (0, pad))

    x = x.reshape(-1, x.shape[-1] // block_size, block_size)
    xmax = torch.amax(torch.abs(x), dim=-1, keepdim=True) * clip_ratio
    safe_xmax = torch.where(xmax == 0, torch.ones_like(xmax), xmax)
    scale = 2 ** torch.ceil(torch.log2(safe_xmax / maxq))
    scale = torch.clamp(scale, min=finfo.tiny, max=finfo.max)
    scale = torch.where(xmax == 0, torch.ones_like(scale), scale)

    q = torch.clamp(torch.round(x / scale), minq, maxq)
    q = torch.nan_to_num(q, nan=0.0, posinf=maxq, neginf=minq)
    xhat = (q * scale).reshape(*orig_shape[:-1], -1)
    xhat = torch.nan_to_num(xhat, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    xhat = torch.clamp(xhat, min=orig_finfo.min, max=orig_finfo.max)

    if pad:
        xhat = xhat[..., : orig_shape[-1]]
    return xhat.reshape(orig_shape).to(dtype=orig_dtype)


def maybe_bfp_fake_quant(x, bits=4, block_size=BFP_DEFAULT_BLOCK_SIZE, clip_ratio=1.0):
    if bits is None or bits <= 0:
        return x
    return bfp_fake_quant(x, bits=bits, block_size=block_size, clip_ratio=clip_ratio)


def _bfp_input_pre_hook(bits, block_size, clip_ratio):
    def hook(module, inputs):
        if len(inputs) == 0:
            return inputs
        return (
            bfp_fake_quant(
                inputs[0],
                bits=bits,
                block_size=block_size,
                clip_ratio=clip_ratio,
            ),
            *inputs[1:],
        )

    return hook


def _bfp_output_hook(bits, block_size, clip_ratio):
    def hook(module, inputs, output):
        return bfp_fake_quant(
            output,
            bits=bits,
            block_size=block_size,
            clip_ratio=clip_ratio,
        )

    return hook


def _remove_hook(module, attr_name):
    handle = getattr(module, attr_name, None)
    if handle is not None:
        handle.remove()
        delattr(module, attr_name)


def add_activation_bfp_to_linear(module, bits=4, groupsize=-1, clip_ratio=1.0):
    block_size = resolve_bfp_block_size(groupsize)
    _remove_hook(module, "_bfp_input_handle")
    module._bfp_input_handle = module.register_forward_pre_hook(
        _bfp_input_pre_hook(bits, block_size, clip_ratio)
    )
    module.bfp_input_bits = bits
    module.bfp_input_block_size = block_size
    module.bfp_input_clip_ratio = clip_ratio


def add_output_bfp_to_linear(module, bits=4, groupsize=-1, clip_ratio=1.0):
    block_size = resolve_bfp_block_size(groupsize)
    _remove_hook(module, "_bfp_output_handle")
    module._bfp_output_handle = module.register_forward_hook(
        _bfp_output_hook(bits, block_size, clip_ratio)
    )
    module.bfp_output_bits = bits
    module.bfp_output_block_size = block_size
    module.bfp_output_clip_ratio = clip_ratio


def add_activation_bfp_to_llama(model, bits=4, groupsize=-1, clip_ratio=1.0):
    lm_head = getattr(model, "lm_head", None)
    wrapped = 0

    for module in model.modules():
        if module is lm_head:
            continue
        if isinstance(module, torch.nn.Linear):
            add_activation_bfp_to_linear(
                module,
                bits=bits,
                groupsize=groupsize,
                clip_ratio=clip_ratio,
            )
            wrapped += 1

    return wrapped


def add_v_bfp_to_llama(model, bits=4, groupsize=-1, clip_ratio=1.0):
    wrapped = 0

    for layer in model.model.layers:
        add_output_bfp_to_linear(
            layer.self_attn.v_proj,
            bits=bits,
            groupsize=groupsize,
            clip_ratio=clip_ratio,
        )
        wrapped += 1

    return wrapped


def _build_k_bfp_forward(attn, bits, block_size, clip_ratio, qk_online_had):
    if apply_rotary_pos_emb is None or repeat_kv is None:
        raise ImportError("K BFP requires transformers LLaMA attention helpers.")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp,
                dim=0,
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if qk_online_had:
            query_states = apply_hadamard_to_last_dim(query_states, block_size)
            key_states = apply_hadamard_to_last_dim(key_states, block_size)
        key_states = maybe_bfp_fake_quant(
            key_states,
            bits=bits,
            block_size=block_size,
            clip_ratio=clip_ratio,
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states.float(),
            key_states.float().transpose(2, 3),
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask.to(dtype=attn_weights.dtype)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=1.0, neginf=0.0)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states.float())

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = clamp_to_dtype_range(attn_output, self.o_proj.weight.dtype)
        attn_output = torch.nan_to_num(
            attn_output,
            nan=0.0,
            posinf=torch.finfo(self.o_proj.weight.dtype).max,
            neginf=torch.finfo(self.o_proj.weight.dtype).min,
        ).to(dtype=self.o_proj.weight.dtype)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return types.MethodType(forward, attn)


def add_k_bfp_to_llama(model, bits=4, groupsize=-1, clip_ratio=1.0, qk_online_had=True):
    block_size = resolve_bfp_block_size(groupsize)
    wrapped = 0

    for layer in model.model.layers:
        attn = layer.self_attn
        if not hasattr(attn, "_original_forward"):
            attn._original_forward = attn.forward
        attn.forward = _build_k_bfp_forward(attn, bits, block_size, clip_ratio, qk_online_had)
        attn.bfp_k_bits = bits
        attn.bfp_k_block_size = block_size
        attn.bfp_k_clip_ratio = clip_ratio
        attn.bfp_k_qk_online_had = qk_online_had
        wrapped += 1

    return wrapped


def add_bfp_to_llama(
    model,
    *,
    a_bits=None,
    a_groupsize=-1,
    a_clip_ratio=1.0,
    v_bits=None,
    v_groupsize=-1,
    v_clip_ratio=1.0,
    k_bits=None,
    k_groupsize=-1,
    k_clip_ratio=1.0,
    qk_online_had=True,
    force_qk_online_had=False,
):
    counts = {"activation": 0, "v": 0, "k": 0}

    if a_bits is not None:
        counts["activation"] = add_activation_bfp_to_llama(
            model,
            bits=a_bits,
            groupsize=a_groupsize,
            clip_ratio=a_clip_ratio,
        )
    if v_bits is not None:
        counts["v"] = add_v_bfp_to_llama(
            model,
            bits=v_bits,
            groupsize=v_groupsize,
            clip_ratio=v_clip_ratio,
        )
    if k_bits is not None or force_qk_online_had:
        counts["k"] = add_k_bfp_to_llama(
            model,
            bits=k_bits,
            groupsize=k_groupsize,
            clip_ratio=k_clip_ratio,
            qk_online_had=qk_online_had,
        )

    return counts
