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
    quantize_weight=True,
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

        if quantize_weight:
            W_b_quant = quant_bfp_mantissa(
                W_b_corrected.t(),
                block_size=bs,
                mantissa_bits=mantissa_bits,
            ).t()
        else:
            W_b_quant = W_b_corrected
        W_Q[:, start:end] = W_b_quant[:, inv_order]

    return W_Q.to(dtype=weight.dtype)


def _iter_target_linears(model, include_lm_head=False):
    lm_head = getattr(model, "lm_head", None)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if not include_lm_head and module is lm_head:
                continue
            yield name, module


def _iter_layer_target_linears(layer, layer_name):
    for name, module in layer.named_modules():
        if isinstance(module, torch.nn.Linear):
            full_name = f"{layer_name}.{name}" if name else layer_name
            yield full_name, module


def _get_llama_base_model(model):
    return getattr(model, "model", model)


def _module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _init_stats(in_features, block_size, device):
    blocks = []
    for start in range(0, in_features, block_size):
        end = min(start + block_size, in_features)
        bs = end - start
        blocks.append(
            {
                "cross": torch.zeros(bs, bs, device=device, dtype=torch.float32),
                "hessian": torch.zeros(bs, bs, device=device, dtype=torch.float32),
                "magnitude": torch.zeros(bs, device=device, dtype=torch.float32),
                "count": 0,
            }
        )
    return {"in_features": in_features, "block_size": block_size, "blocks": blocks}


def _make_stats_hook(name, store, block_size, mantissa_bits):
    def hook(module, inputs):
        if len(inputs) == 0:
            return
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
        if name not in store:
            store[name] = _init_stats(x.shape[1], block_size, x.device)

        stats = store[name]
        for block_idx, start in enumerate(range(0, x.shape[1], block_size)):
            end = min(start + block_size, x.shape[1])
            X_b = x[:, start:end].t().contiguous()
            bs = end - start
            X_Q_b = quant_bfp_mantissa(X_b.t(), block_size=bs, mantissa_bits=mantissa_bits).t()
            block = stats["blocks"][block_idx]
            block["cross"] += (X_b - X_Q_b) @ X_Q_b.t()
            block["hessian"] += 2 * (X_Q_b @ X_Q_b.t())
            block["magnitude"] += X_b.abs().sum(dim=1)
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


def _load_calibration_input_ids(tokenizer, *, split, seqlen, nsamples, seed):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    starts = sample_calibration_starts(
        input_ids.shape[1],
        seqlen=seqlen,
        nsamples=nsamples,
        seed=seed,
    )
    return input_ids, starts


def _llama_layer_kwargs(base_model, hidden_states):
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = None
    if hasattr(base_model, "_update_causal_mask"):
        causal_mask = base_model._update_causal_mask(
            None,
            hidden_states,
            cache_position,
            None,
            False,
        )

    kwargs = {
        "attention_mask": causal_mask,
        "position_ids": position_ids,
        "past_key_value": None,
        "output_attentions": False,
        "use_cache": False,
        "cache_position": cache_position,
    }
    if hasattr(base_model, "rotary_emb"):
        kwargs["position_embeddings"] = base_model.rotary_emb(hidden_states, position_ids)
    return kwargs


def _run_decoder_layer(base_model, layer, hidden_states):
    layer_device = _module_device(layer)
    hidden_states = hidden_states.to(device=layer_device)
    outputs = layer(hidden_states, **_llama_layer_kwargs(base_model, hidden_states))
    return outputs[0]


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
def apply_bfp_gptq_stats_to_targets(
    targets,
    stats,
    *,
    mantissa_bits=5,
    lambda_reg=1e-4,
    quantize_weight=True,
):
    corrected = 0
    for name, module in targets:
        if name not in stats:
            continue
        module.weight.data = correct_and_quantize_weight_bfp_gptq_from_stats(
            module.weight.data,
            stats[name],
            mantissa_bits=mantissa_bits,
            lambda_reg=lambda_reg,
            quantize_weight=quantize_weight,
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
    quantize_weight=True,
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
        if quantize_weight:
            W_b_quant = quant_bfp_mantissa(
                W_b_corrected.t(),
                block_size=bs,
                mantissa_bits=mantissa_bits,
            ).t()
        else:
            W_b_quant = W_b_corrected
        W_Q[:, start:end] = W_b_quant[:, inv_order]

    return W_Q.to(dtype=weight.dtype)


@torch.no_grad()
def apply_bfp_gptq_stats_to_llama(
    model,
    stats,
    *,
    mantissa_bits=5,
    lambda_reg=1e-4,
    quantize_weight=True,
    include_lm_head=False,
    show_progress=True,
):
    targets = dict(_iter_target_linears(model, include_lm_head=include_lm_head))
    items = [(name, module) for name, module in targets.items() if name in stats]
    iterator = tqdm(items, desc="BFP-GPTQ weights", disable=not show_progress)

    corrected = 0
    for name, module in iterator:
        corrected += apply_bfp_gptq_stats_to_targets(
            [(name, module)],
            stats,
            mantissa_bits=mantissa_bits,
            lambda_reg=lambda_reg,
            quantize_weight=quantize_weight,
        )

    return corrected


@torch.no_grad()
def calibrate_and_apply_bfp_gptq_to_llama_layerwise(
    model,
    tokenizer,
    *,
    calib_split="train",
    calib_samples=128,
    calib_seqlen=2048,
    calib_seed=0,
    block_size=32,
    mantissa_bits=5,
    lambda_reg=1e-4,
    quantize_weight=True,
    show_progress=True,
):
    base_model = _get_llama_base_model(model)
    if not hasattr(base_model, "layers") or not hasattr(base_model, "embed_tokens"):
        raise ValueError("Layer-wise BFP-GPTQ expects a LLaMA-style model with layers/embed_tokens.")

    was_training = model.training
    original_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if original_use_cache is not None:
        model.config.use_cache = False

    try:
        input_ids, starts = _load_calibration_input_ids(
            tokenizer,
            split=calib_split,
            seqlen=calib_seqlen,
            nsamples=calib_samples,
            seed=calib_seed,
        )

        embed_device = _module_device(base_model.embed_tokens)
        hidden_states = []
        sample_iter = tqdm(starts, desc="BFP-GPTQ embeddings", disable=not show_progress)
        for start in sample_iter:
            ids = input_ids[:, start : start + calib_seqlen].to(embed_device)
            hidden_states.append(base_model.embed_tokens(ids).detach().cpu())

        layer_prefix = "model.layers" if getattr(model, "model", None) is base_model else "layers"
        corrected = 0
        layer_iter = tqdm(
            enumerate(base_model.layers),
            total=len(base_model.layers),
            desc="BFP-GPTQ layers",
            disable=not show_progress,
        )
        for layer_idx, layer in layer_iter:
            layer_name = f"{layer_prefix}.{layer_idx}"
            layer_iter.set_postfix_str(layer_name)
            targets = list(_iter_layer_target_linears(layer, layer_name))
            stats = {}
            handles = [
                module.register_forward_pre_hook(_make_stats_hook(name, stats, block_size, mantissa_bits))
                for name, module in targets
            ]
            try:
                stat_iter = tqdm(
                    hidden_states,
                    desc=f"{layer_name} stats",
                    leave=False,
                    disable=not show_progress,
                )
                for hidden in stat_iter:
                    _run_decoder_layer(base_model, layer, hidden)
            finally:
                for handle in handles:
                    handle.remove()

            corrected += apply_bfp_gptq_stats_to_targets(
                targets,
                stats,
                mantissa_bits=mantissa_bits,
                lambda_reg=lambda_reg,
                quantize_weight=quantize_weight,
            )

            next_hidden_states = []
            output_iter = tqdm(
                hidden_states,
                desc=f"{layer_name} outputs",
                leave=False,
                disable=not show_progress,
            )
            for hidden in output_iter:
                next_hidden_states.append(_run_decoder_layer(base_model, layer, hidden).detach().cpu())
            hidden_states = next_hidden_states

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        if was_training:
            model.train()

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
    block_size=32,
    mantissa_bits=5,
    lambda_reg=1e-4,
    quantize_weight=True,
    include_lm_head=False,
    show_progress=True,
    calib_mode="layerwise",
):
    if calib_mode == "layerwise":
        if include_lm_head:
            raise ValueError("Layer-wise BFP-GPTQ does not support include_lm_head=True.")
        return calibrate_and_apply_bfp_gptq_to_llama_layerwise(
            model,
            tokenizer,
            calib_split=calib_split,
            calib_samples=calib_samples,
            calib_seqlen=calib_seqlen,
            calib_seed=calib_seed,
            block_size=block_size,
            mantissa_bits=mantissa_bits,
            lambda_reg=lambda_reg,
            quantize_weight=quantize_weight,
            show_progress=show_progress,
        )
    if calib_mode != "global":
        raise ValueError(f"Unknown BFP-GPTQ calibration mode: {calib_mode}.")

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
        quantize_weight=quantize_weight,
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
