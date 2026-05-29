import torch
from hadamard_utils import matmul_hadU, random_hadamard_matrix


def _is_pow2(value):
    return value > 0 and value & (value - 1) == 0


def _validate_block_size(size, block_size):
    if block_size is None:
        if not _is_pow2(size):
            raise ValueError(f"Full Hadamard requires a power-of-2 dimension, got {size}.")
        return
    if not _is_pow2(block_size):
        raise ValueError(f"Hadamard block size must be a power of 2, got {block_size}.")
    if size % block_size != 0:
        raise ValueError(f"Dimension {size} must be divisible by block size {block_size}.")


def resolve_rotation_block_size(rotation_block_size, hidden_size, bfp_block_size=32):
    if rotation_block_size == -1:
        block_size = bfp_block_size
    elif rotation_block_size == 0:
        block_size = None
    else:
        block_size = rotation_block_size

    _validate_block_size(hidden_size, block_size)
    return block_size


def normalized_hadamard(size, *, device=None, dtype=None):
    if not _is_pow2(size):
        raise ValueError(f"Hadamard size must be a power of 2, got {size}.")

    eye = torch.eye(size, device=device, dtype=dtype)
    return matmul_hadU(eye)


def _random_hadamard_matrix(size, *, block_size=None, seed=0, device=None, dtype=None):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        rotation = random_hadamard_matrix(size, device=device, block_size=block_size)
    if dtype is not None:
        rotation = rotation.to(dtype=dtype)
    return rotation


def build_rotation_matrix(
    size,
    *,
    block_size=None,
    seed=0,
    device=None,
    dtype=None,
):
    _validate_block_size(size, block_size)

    dtype = dtype or torch.float32
    return _random_hadamard_matrix(
        size,
        block_size=block_size,
        seed=seed,
        device=device,
        dtype=dtype,
    )


def apply_rotation_to_last_dim(x, block_size=None, seed=0):
    dim = x.shape[-1]
    _validate_block_size(dim, block_size)

    if block_size is None:
        block_size = dim

    compute_dtype = _compute_dtype(x.dtype)
    x_dtype = x.dtype
    x = x.to(dtype=compute_dtype)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        signs = torch.randint(0, 2, (dim,), dtype=torch.int64)
    signs = signs.to(device=x.device, dtype=compute_dtype).mul_(2).sub_(1)
    signs = signs.reshape(dim // block_size, block_size)

    x_blocks = x.reshape(*x.shape[:-1], dim // block_size, block_size)
    output = matmul_hadU(x_blocks * signs)
    return output.reshape_as(x).to(dtype=x_dtype)


def build_online_hadamard(size, *, block_size=None, device=None, dtype=None):
    _validate_block_size(size, block_size)

    if block_size is None:
        return normalized_hadamard(size, device=device, dtype=dtype)

    block = normalized_hadamard(block_size, device=device, dtype=dtype)
    hadamard = torch.zeros(size, size, device=device, dtype=dtype)
    for start in range(0, size, block_size):
        hadamard[start : start + block_size, start : start + block_size] = block
    return hadamard


def _compute_dtype(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


@torch.no_grad()
def rotate_weight_input(module, rotation):
    dtype = module.weight.dtype
    compute_dtype = _compute_dtype(dtype)
    weight = module.weight.data.to(dtype=compute_dtype)
    q = rotation.to(device=weight.device, dtype=compute_dtype)
    module.weight.data = (weight @ q).to(dtype=dtype)


@torch.no_grad()
def rotate_weight_output(module, rotation):
    dtype = module.weight.dtype
    compute_dtype = _compute_dtype(dtype)
    weight = module.weight.data.to(dtype=compute_dtype)
    q = rotation.to(device=weight.device, dtype=compute_dtype)
    module.weight.data = (q.t() @ weight).to(dtype=dtype)

    if module.bias is not None:
        bias_dtype = module.bias.dtype
        bias = module.bias.data.to(dtype=compute_dtype)
        module.bias.data = (q.t() @ bias).to(dtype=bias_dtype)


@torch.no_grad()
def rotate_weight_input_random_hadamard(module, block_size=None, seed=0):
    module.weight.data = apply_rotation_to_last_dim(module.weight.data, block_size, seed)


@torch.no_grad()
def rotate_weight_output_random_hadamard(module, block_size=None, seed=0):
    module.weight.data = apply_rotation_to_last_dim(
        module.weight.data.t(),
        block_size,
        seed,
    ).t()

    if module.bias is not None:
        module.bias.data = apply_rotation_to_last_dim(module.bias.data, block_size, seed)


def apply_hadamard_to_last_dim(x, block_size=None):
    dim = x.shape[-1]
    _validate_block_size(dim, block_size)

    compute_dtype = _compute_dtype(x.dtype)
    x_dtype = x.dtype
    x = x.to(dtype=compute_dtype)

    if block_size is None:
        return matmul_hadU(x).to(dtype=x_dtype)

    output = matmul_hadU(x.reshape(*x.shape[:-1], dim // block_size, block_size))
    return output.reshape_as(x).to(dtype=x_dtype)


def apply_headwise_hadamard_to_last_dim(x, num_heads, head_dim, block_size):
    dim = x.shape[-1]
    if dim != num_heads * head_dim:
        raise ValueError(
            "Last dimension must equal num_heads * head_dim: "
            f"{dim} != {num_heads} * {head_dim}."
        )
    _validate_block_size(head_dim, block_size)

    compute_dtype = _compute_dtype(x.dtype)
    x_dtype = x.dtype
    x = x.to(dtype=compute_dtype)

    if block_size is None:
        output = matmul_hadU(x.reshape(*x.shape[:-1], num_heads, head_dim))
    else:
        output = matmul_hadU(
            x.reshape(*x.shape[:-1], num_heads, head_dim // block_size, block_size)
        )
    return output.reshape_as(x).to(dtype=x_dtype)


def _online_hadamard_pre_hook(block_size):
    def hook(module, inputs):
        if len(inputs) == 0:
            return inputs
        return (apply_hadamard_to_last_dim(inputs[0], block_size), *inputs[1:])

    return hook


def _headwise_online_hadamard_pre_hook(num_heads, head_dim, block_size):
    def hook(module, inputs):
        if len(inputs) == 0:
            return inputs
        return (
            apply_headwise_hadamard_to_last_dim(
                inputs[0],
                num_heads,
                head_dim,
                block_size,
            ),
            *inputs[1:],
        )

    return hook


@torch.no_grad()
def add_online_hadamard_to_linear(module, block_size=None):
    if hasattr(module, "_online_hadamard_handle"):
        module._online_hadamard_handle.remove()

    module.weight.data = apply_hadamard_to_last_dim(module.weight.data, block_size)

    module._online_hadamard_handle = module.register_forward_pre_hook(
        _online_hadamard_pre_hook(block_size)
    )
    module.online_full_had = block_size is None
    module.online_had_block_size = block_size


@torch.no_grad()
def add_headwise_online_hadamard_to_linear(module, num_heads, head_dim, block_size=None):
    if hasattr(module, "_online_hadamard_handle"):
        module._online_hadamard_handle.remove()

    module.weight.data = apply_headwise_hadamard_to_last_dim(
        module.weight.data,
        num_heads,
        head_dim,
        block_size,
    )

    module._online_hadamard_handle = module.register_forward_pre_hook(
        _headwise_online_hadamard_pre_hook(num_heads, head_dim, block_size)
    )
    module.online_full_had = block_size is None
    module.online_had_block_size = block_size
    module.online_had_num_heads = num_heads
    module.online_had_head_dim = head_dim


@torch.no_grad()
def rotate_llama_model(
    model,
    *,
    rotation_block_size=32,
    seed=0,
    bfp_block_size=32,
    online_down_proj_had=True,
    online_o_proj_had=True,
):
    hidden_size = model.config.hidden_size
    block_size = resolve_rotation_block_size(rotation_block_size, hidden_size, bfp_block_size)
    num_heads = getattr(model.config, "num_attention_heads", None)
    if num_heads is None:
        raise ValueError("model.config.num_attention_heads is required for o_proj rotation.")
    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads.")
    head_dim = hidden_size // num_heads

    rotate_weight_input_random_hadamard(model.model.embed_tokens, block_size, seed)

    for layer in model.model.layers:
        rotate_weight_input_random_hadamard(layer.self_attn.q_proj, block_size, seed)
        rotate_weight_input_random_hadamard(layer.self_attn.k_proj, block_size, seed)
        rotate_weight_input_random_hadamard(layer.self_attn.v_proj, block_size, seed)
        rotate_weight_output_random_hadamard(layer.self_attn.o_proj, block_size, seed)

        rotate_weight_input_random_hadamard(layer.mlp.up_proj, block_size, seed)
        rotate_weight_input_random_hadamard(layer.mlp.gate_proj, block_size, seed)
        rotate_weight_output_random_hadamard(layer.mlp.down_proj, block_size, seed)

        if online_o_proj_had:
            if block_size is None:
                add_online_hadamard_to_linear(layer.self_attn.o_proj, block_size)
            else:
                add_headwise_online_hadamard_to_linear(
                    layer.self_attn.o_proj,
                    num_heads,
                    head_dim,
                    block_size,
                )
        if online_down_proj_had:
            add_online_hadamard_to_linear(layer.mlp.down_proj, block_size)

    rotate_weight_input_random_hadamard(model.lm_head, block_size, seed)
    return model
