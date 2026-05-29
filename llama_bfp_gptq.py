import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm


BFP_GPTQ_DEFAULT_BLOCK_SIZE = 32


def quant_bfp_mantissa(x, block_size=32, mantissa_bits=5):
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if mantissa_bits <= 0:
        raise ValueError(f"mantissa_bits must be positive, got {mantissa_bits}.")

    orig_shape = x.shape
    orig_dtype = x.dtype
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x = x.to(dtype=compute_dtype)

    pad = (block_size - x.shape[-1] % block_size) % block_size
    if pad:
        x = F.pad(x, (0, pad))

    x = x.reshape(-1, block_size)
    max_val = x.abs().amax(dim=1, keepdim=True).clamp(min=torch.finfo(compute_dtype).tiny)
    shared_exp = torch.floor(torch.log2(max_val))
    scale = 2 ** (shared_exp - (mantissa_bits - 1))
    x_q = torch.round(x / scale) * scale

    x_q = x_q.reshape(*orig_shape[:-1], -1)
    if pad:
        x_q = x_q[..., : orig_shape[-1]]
    return x_q.reshape(orig_shape).to(dtype=orig_dtype)


def cholesky_inverse_stable(H, lambda_reg=1e-4):
    eye = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    H_reg = H + lambda_reg * eye
    try:
        L = torch.linalg.cholesky(H_reg)
    except torch.linalg.LinAlgError:
        jitter = lambda_reg * 10
        L = torch.linalg.cholesky(H + jitter * eye)
    return torch.cholesky_inverse(L)


@torch.no_grad()
def correct_and_quantize_weight_bfp_gptq(
    weight,
    activations,
    *,
    block_size=32,
    mantissa_bits=5,
    lambda_reg=1e-4,
):
    """Correct activation-BFP error into weight, then BFP-quantize weight."""
    out_features, in_features = weight.shape
    if activations.shape[0] != in_features:
        raise ValueError(
            "activations must have shape [in_features, samples]: "
            f"{activations.shape[0]} != {in_features}."
        )

    compute_device = weight.device
    W = weight.detach().to(device=compute_device, dtype=torch.float32)
    X = activations.detach().to(device=compute_device, dtype=torch.float32)
    W_Q = torch.empty_like(W)

    for start in range(0, in_features, block_size):
        end = min(start + block_size, in_features)
        bs = end - start

        W_b = W[:, start:end].clone()
        X_b = X[start:end, :].clone()
        X_Q_b = quant_bfp_mantissa(X_b.t(), block_size=bs, mantissa_bits=mantissa_bits).t()

        magnitude = X_b.abs().mean(dim=1)
        order = torch.argsort(magnitude, descending=True)
        inv_order = torch.argsort(order)
        W_b = W_b[:, order]
        X_b = X_b[order, :]
        X_Q_b = X_Q_b[order, :]

        H = 2 * (X_Q_b @ X_Q_b.t())
        H_inv = cholesky_inverse_stable(H, lambda_reg=lambda_reg)

        act_err = X_b - X_Q_b
        delta_W = W_b @ act_err @ X_Q_b.t() @ H_inv
        W_b_corrected = W_b + delta_W

        W_b_quant = quant_bfp_mantissa(
            W_b_corrected.t(),
            block_size=bs,
            mantissa_bits=mantissa_bits,
        ).t()
        W_Q[:, start:end] = W_b_quant[:, inv_order]

    return W_Q.to(dtype=weight.dtype)


def _iter_target_linears(model, include_lm_head=False):
    lm_head = getattr(model, "lm_head", None)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if not include_lm_head and module is lm_head:
                continue
            yield name, module


def _make_capture_hook(name, store, max_samples_per_layer):
    def hook(module, inputs):
        if len(inputs) == 0:
            return
        x = inputs[0].detach()
        x = x.reshape(-1, x.shape[-1]).float().cpu()
        if name in store:
            remaining = max_samples_per_layer - store[name].shape[0]
            if remaining <= 0:
                return
            store[name] = torch.cat([store[name], x[:remaining]], dim=0)
        else:
            store[name] = x[:max_samples_per_layer]

    return hook


def _init_stats(in_features, block_size):
    blocks = []
    for start in range(0, in_features, block_size):
        end = min(start + block_size, in_features)
        bs = end - start
        blocks.append(
            {
                "cross": torch.zeros(bs, bs, dtype=torch.float64),
                "hessian": torch.zeros(bs, bs, dtype=torch.float64),
                "magnitude": torch.zeros(bs, dtype=torch.float64),
                "count": 0,
            }
        )
    return {"in_features": in_features, "block_size": block_size, "blocks": blocks}


def _make_stats_hook(name, store, block_size, mantissa_bits):
    def hook(module, inputs):
        if len(inputs) == 0:
            return
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float().cpu()
        if name not in store:
            store[name] = _init_stats(x.shape[1], block_size)

        stats = store[name]
        for block_idx, start in enumerate(range(0, x.shape[1], block_size)):
            end = min(start + block_size, x.shape[1])
            X_b = x[:, start:end].t().contiguous()
            bs = end - start
            X_Q_b = quant_bfp_mantissa(X_b.t(), block_size=bs, mantissa_bits=mantissa_bits).t()
            block = stats["blocks"][block_idx]
            block["cross"] += ((X_b - X_Q_b) @ X_Q_b.t()).double()
            block["hessian"] += (2 * (X_Q_b @ X_Q_b.t())).double()
            block["magnitude"] += X_b.abs().sum(dim=1).double()
            block["count"] += X_b.shape[1]

    return hook


def sample_calibration_starts(total_len, seqlen=2048, nsamples=128, seed=0):
    if total_len < seqlen:
        raise ValueError(
            f"Tokenized calibration split has {total_len} tokens, "
            f"which is shorter than seqlen={seqlen}."
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randint(
        low=0,
        high=total_len - seqlen + 1,
        size=(nsamples,),
        generator=generator,
    ).tolist()


@torch.no_grad()
def collect_llama_bfp_gptq_stats(
    model,
    tokenizer,
    *,
    split="train",
    nsamples=128,
    seqlen=2048,
    seed=0,
    block_size=32,
    mantissa_bits=5,
    include_lm_head=False,
    show_progress=True,
):
    targets = dict(_iter_target_linears(model, include_lm_head=include_lm_head))
    stats = {}
    handles = [
        module.register_forward_pre_hook(_make_stats_hook(name, stats, block_size, mantissa_bits))
        for name, module in targets.items()
    ]

    was_training = model.training
    original_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if original_use_cache is not None:
        model.config.use_cache = False

    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(dataset["text"])
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        device = next(model.parameters()).device
        starts = sample_calibration_starts(
            input_ids.shape[1],
            seqlen=seqlen,
            nsamples=nsamples,
            seed=seed,
        )
        iterator = tqdm(starts, desc="BFP-GPTQ calibration", disable=not show_progress)
        for start in iterator:
            model(input_ids[:, start : start + seqlen].to(device))
    finally:
        for handle in handles:
            handle.remove()
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        if was_training:
            model.train()

    return stats


@torch.no_grad()
def collect_llama_linear_inputs(
    model,
    tokenizer,
    *,
    split="train",
    nsamples=128,
    seqlen=2048,
    seed=0,
    max_samples_per_layer=4096,
    include_lm_head=False,
    show_progress=True,
):
    targets = dict(_iter_target_linears(model, include_lm_head=include_lm_head))
    captured = {}
    handles = [
        module.register_forward_pre_hook(_make_capture_hook(name, captured, max_samples_per_layer))
        for name, module in targets.items()
    ]

    was_training = model.training
    original_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if original_use_cache is not None:
        model.config.use_cache = False

    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(dataset["text"])
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        device = next(model.parameters()).device
        total_len = input_ids.shape[1]
        starts = sample_calibration_starts(total_len, seqlen=seqlen, nsamples=nsamples, seed=seed)
        iterator = tqdm(starts, desc="BFP-GPTQ calibration", disable=not show_progress)
        for start in iterator:
            end = start + seqlen
            model(input_ids[:, start:end].to(device))
    finally:
        for handle in handles:
            handle.remove()
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        if was_training:
            model.train()

    return {name: value.t().contiguous() for name, value in captured.items()}


@torch.no_grad()
def apply_bfp_gptq_to_llama(
    model,
    activations,
    *,
    block_size=32,
    mantissa_bits=5,
    lambda_reg=1e-4,
    include_lm_head=False,
    show_progress=True,
):
    targets = dict(_iter_target_linears(model, include_lm_head=include_lm_head))
    items = [(name, module) for name, module in targets.items() if name in activations]
    iterator = tqdm(items, desc="BFP-GPTQ weights", disable=not show_progress)

    corrected = 0
    for name, module in iterator:
        module.weight.data = correct_and_quantize_weight_bfp_gptq(
            module.weight.data,
            activations[name],
            block_size=block_size,
            mantissa_bits=mantissa_bits,
            lambda_reg=lambda_reg,
        )
        corrected += 1

    return corrected


@torch.no_grad()
def correct_and_quantize_weight_bfp_gptq_from_stats(
    weight,
    stats,
    *,
    mantissa_bits=5,
    lambda_reg=1e-4,
):
    out_features, in_features = weight.shape
    if stats["in_features"] != in_features:
        raise ValueError(f"Stats in_features mismatch: {stats['in_features']} != {in_features}.")

    block_size = stats["block_size"]
    W = weight.detach().float()
    W_Q = torch.empty_like(W)

    for block_idx, start in enumerate(range(0, in_features, block_size)):
        end = min(start + block_size, in_features)
        bs = end - start
        block = stats["blocks"][block_idx]

        cross = block["cross"].to(device=W.device, dtype=torch.float32)
        H = block["hessian"].to(device=W.device, dtype=torch.float32)
        magnitude = block["magnitude"].to(device=W.device, dtype=torch.float32)

        order = torch.argsort(magnitude, descending=True)
        inv_order = torch.argsort(order)
        W_b = W[:, start:end][:, order]
        cross = cross[order][:, order]
        H = H[order][:, order]

        H_inv = cholesky_inverse_stable(H, lambda_reg=lambda_reg)
        W_b_corrected = W_b + W_b @ cross @ H_inv
        W_b_quant = quant_bfp_mantissa(
            W_b_corrected.t(),
            block_size=bs,
            mantissa_bits=mantissa_bits,
        ).t()
        W_Q[:, start:end] = W_b_quant[:, inv_order]

    return W_Q.to(dtype=weight.dtype)


@torch.no_grad()
def apply_bfp_gptq_stats_to_llama(
    model,
    stats,
    *,
    mantissa_bits=5,
    lambda_reg=1e-4,
    include_lm_head=False,
    show_progress=True,
):
    targets = dict(_iter_target_linears(model, include_lm_head=include_lm_head))
    items = [(name, module) for name, module in targets.items() if name in stats]
    iterator = tqdm(items, desc="BFP-GPTQ weights", disable=not show_progress)

    corrected = 0
    for name, module in iterator:
        module.weight.data = correct_and_quantize_weight_bfp_gptq_from_stats(
            module.weight.data,
            stats[name],
            mantissa_bits=mantissa_bits,
            lambda_reg=lambda_reg,
        )
        corrected += 1

    return corrected


@torch.no_grad()
def calibrate_and_apply_bfp_gptq_to_llama(
    model,
    tokenizer,
    *,
    calib_split="train",
    calib_samples=128,
    calib_seqlen=2048,
    calib_seed=0,
    max_samples_per_layer=None,
    block_size=32,
    mantissa_bits=5,
    lambda_reg=1e-4,
    include_lm_head=False,
    show_progress=True,
):
    stats = collect_llama_bfp_gptq_stats(
        model,
        tokenizer,
        split=calib_split,
        nsamples=calib_samples,
        seqlen=calib_seqlen,
        seed=calib_seed,
        block_size=block_size,
        mantissa_bits=mantissa_bits,
        include_lm_head=include_lm_head,
        show_progress=show_progress,
    )
    return apply_bfp_gptq_stats_to_llama(
        model,
        stats,
        mantissa_bits=mantissa_bits,
        lambda_reg=lambda_reg,
        include_lm_head=include_lm_head,
        show_progress=show_progress,
    )


def get_bfp_gptq_weight_state(model, *, include_lm_head=False):
    return {
        name: module.weight.detach().cpu()
        for name, module in _iter_target_linears(model, include_lm_head=include_lm_head)
    }


def save_bfp_gptq_weights(model, path, *, metadata=None, include_lm_head=False):
    payload = {
        "metadata": metadata or {},
        "weights": get_bfp_gptq_weight_state(model, include_lm_head=include_lm_head),
    }
    torch.save(payload, path)


def load_bfp_gptq_weights(model, path, *, strict=True):
    payload = torch.load(path, map_location="cpu")
    weights = payload["weights"] if isinstance(payload, dict) and "weights" in payload else payload
    modules = dict(model.named_modules())
    loaded = 0

    for name, weight in weights.items():
        if name not in modules or not isinstance(modules[name], torch.nn.Linear):
            if strict:
                raise KeyError(f"No Linear module named {name!r} in model.")
            continue
        module = modules[name]
        if tuple(module.weight.shape) != tuple(weight.shape):
            if strict:
                raise ValueError(
                    f"Shape mismatch for {name}: {tuple(module.weight.shape)} != {tuple(weight.shape)}."
                )
            continue
        module.weight.data.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
        loaded += 1

    return loaded
