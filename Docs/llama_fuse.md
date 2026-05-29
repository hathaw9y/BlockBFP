# LLaMA Model Fuse Notes

This note describes only the LLaMA path used by `fake_quant/rotation_utils.py`.
It is meant as an implementation guide for rewriting the fuse step from scratch.

## Goal

QuaRot removes the learned scale from LLaMA RMSNorm layers by folding it into the
following Linear weights. After that, each `LlamaRMSNorm` is replaced with an
unweighted RMS normalization (`RMSN`).

For a LLaMA RMSNorm followed by a Linear:

```text
y = RMSNorm(x) = x / rms(x) * gamma
z = Linear(y) = y @ W.T
```

Fold `gamma` into the Linear weight:

```text
W_fused[:, i] = W[:, i] * gamma[i]
z = (x / rms(x)) @ W_fused.T
```

In PyTorch, Linear weight shape is `[out_features, in_features]`, so this is:

```python
linear.weight.data = linear.weight.data * layernorm.weight
```

The multiplication broadcasts over the input-feature dimension.

## LLaMA Fuse Targets

For each decoder layer:

1. `layer.input_layernorm` is fused into:

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
```

2. `layer.post_attention_layernorm` is fused into:

```text
mlp.up_proj
mlp.gate_proj
```

3. Final model norm `model.model.norm` is fused into:

```text
model.lm_head
```

There is no LLaMA LayerNorm bias in the normal path. The generic helper has a
bias branch, but for standard LLaMA RMSNorm it is not used.

## Embedding Mean Centering

Before fusing layer norms, QuaRot centers the token embedding rows:

```python
embed.weight = embed.weight - embed.weight.mean(dim=-1, keepdim=True)
```

For LLaMA this applies to:

```text
model.model.embed_tokens
```

This is not folding a LayerNorm scale. It is a separate preprocessing step used
by QuaRot before rotations.

## Replace RMSNorm With RMSN

After all learned RMSNorm scales have been folded into adjacent Linear weights,
replace every `transformers.models.llama.modeling_llama.LlamaRMSNorm` with an
unweighted RMS normalization:

```python
class RMSN(torch.nn.Module):
    def __init__(self, mean_dim, eps=1e-5):
        super().__init__()
        self.mean_dim = mean_dim
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)
```

For LLaMA, `mean_dim = model.config.hidden_size`.

## Minimal Implementation Sketch

```python
def fuse_ln_into_linear(layernorm, linear_layers):
    for linear in linear_layers:
        dtype = linear.weight.dtype
        W = linear.weight.data.double()
        gamma = layernorm.weight.data.double()
        linear.weight.data = (W * gamma).to(dtype)


def center_embeddings(model):
    emb = model.model.embed_tokens
    dtype = emb.weight.dtype
    W = emb.weight.data.double()
    emb.weight.data = (W - W.mean(dim=-1, keepdim=True)).to(dtype)


def fuse_llama_model(model):
    center_embeddings(model)

    for layer in model.model.layers:
        fuse_ln_into_linear(
            layer.input_layernorm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        )
        fuse_ln_into_linear(
            layer.post_attention_layernorm,
            [layer.mlp.up_proj, layer.mlp.gate_proj],
        )

    fuse_ln_into_linear(model.model.norm, [model.lm_head])
    replace_all_llama_rmsnorm_with_rmsn(model, model.config.hidden_size)
```

## Important Ordering

Run fuse before applying rotation matrices:

```text
load model
fuse LLaMA RMSNorm scales into weights
replace LLaMA RMSNorm with RMSN
apply QuaRot rotations
add activation quant wrappers
evaluate
```

If rotations are applied before the fuse step, the weight transformations no
longer match the assumptions in the original QuaRot flow.

## What Is Not Fused

The LLaMA fuse step does not fold RMSNorm into:

```text
self_attn.o_proj
mlp.down_proj
```

Those projections are handled later by the rotation logic, not by the RMSNorm
scale-fusion step.
