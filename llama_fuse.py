import torch


try:
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
except ImportError:
    LlamaRMSNorm = None


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


@torch.no_grad()
def fuse_ln_into_linear(layernorm, linear_layers):
    gamma = layernorm.weight.data.double()

    for linear in linear_layers:
        if not isinstance(linear, torch.nn.Linear):
            raise TypeError(f"Expected torch.nn.Linear, got {type(linear).__name__}.")
        if linear.weight.shape[1] != gamma.numel():
            raise ValueError(
                "LayerNorm weight size must match Linear input features: "
                f"{gamma.numel()} != {linear.weight.shape[1]}."
            )

        dtype = linear.weight.dtype
        weight = linear.weight.data.double()
        linear.weight.data = (weight * gamma.to(weight.device)).to(dtype)


@torch.no_grad()
def center_embeddings(model):
    embed_tokens = model.model.embed_tokens
    dtype = embed_tokens.weight.dtype
    weight = embed_tokens.weight.data.double()
    embed_tokens.weight.data = (weight - weight.mean(dim=-1, keepdim=True)).to(dtype)


def _is_llama_rmsnorm(module):
    if LlamaRMSNorm is not None and isinstance(module, LlamaRMSNorm):
        return True
    return module.__class__.__name__ == "LlamaRMSNorm"


def replace_all_llama_rmsnorm_with_rmsn(module, mean_dim):
    replaced = 0

    for name, child in module.named_children():
        if _is_llama_rmsnorm(child):
            eps = getattr(child, "variance_epsilon", getattr(child, "eps", 1e-5))
            setattr(module, name, RMSN(mean_dim, eps=eps))
            replaced += 1
        else:
            replaced += replace_all_llama_rmsnorm_with_rmsn(child, mean_dim)

    return replaced


@torch.no_grad()
def fuse_llama_model(model):
    """Fuse LLaMA RMSNorm weights into adjacent Linear layers in-place."""
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
    return model
