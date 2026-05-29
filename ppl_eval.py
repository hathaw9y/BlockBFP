import torch
from datasets import load_dataset
from tqdm.auto import tqdm


@torch.no_grad()
def evaluate_wikitext2_ppl(
    model,
    tokenizer,
    *,
    split="test",
    max_length=None,
    stride=None,
    device=None,
    max_eval_tokens=None,
    show_progress=True,
):
    """Evaluate causal language-model perplexity on WikiText-2."""
    was_training = model.training
    original_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if original_use_cache is not None:
        model.config.use_cache = False

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model.to(device)

    if max_length is None:
        max_length = getattr(model.config, "max_position_embeddings", None)
        if max_length is None or max_length > 4096:
            max_length = getattr(tokenizer, "model_max_length", 2048)
        if max_length is None or max_length > 100000:
            max_length = 2048

    if stride is None:
        stride = max_length
    if stride > max_length:
        raise ValueError("stride must be less than or equal to max_length.")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    if max_eval_tokens is not None:
        input_ids = input_ids[:, :max_eval_tokens]

    total_nll = 0.0
    total_tokens = 0
    seq_len = input_ids.size(1)
    start_locs = range(0, seq_len, stride)

    progress = tqdm(
        start_locs,
        desc=f"WikiText-2 {split} PPL",
        unit="window",
        disable=not show_progress,
    )
    for start_loc in progress:
        begin_loc = max(start_loc + stride - max_length, 0)
        end_loc = min(start_loc + stride, seq_len)
        trg_len = end_loc - start_loc
        input_ids_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_slice.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids_slice, labels=target_ids)
        valid_tokens = (target_ids[:, 1:] != -100).sum().item()
        if not torch.isfinite(outputs.loss):
            raise FloatingPointError(
                "Non-finite loss during WikiText-2 PPL evaluation at "
                f"window start={start_loc}, end={end_loc}."
            )
        total_nll += outputs.loss.item() * valid_tokens
        total_tokens += valid_tokens

        if total_tokens > 0:
            running_ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()
            progress.set_postfix(ppl=f"{running_ppl:.4f}", tokens=total_tokens)

        if end_loc == seq_len:
            break

    if total_tokens == 0:
        raise ValueError("No tokens were available for WikiText-2 PPL evaluation.")

    ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()

    if was_training:
        model.train()
    if original_use_cache is not None:
        model.config.use_cache = original_use_cache

    return ppl
