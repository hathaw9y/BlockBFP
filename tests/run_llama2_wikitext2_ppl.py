import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import evaluate_wikitext2_ppl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-eval-tokens", type=int, default=1024)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )

    model_kwargs = {
        "local_files_only": args.local_files_only,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    max_eval_tokens = None if args.max_eval_tokens <= 0 else args.max_eval_tokens
    ppl = evaluate_wikitext2_ppl(
        model,
        tokenizer,
        split=args.split,
        max_length=args.max_length,
        stride=args.stride,
        max_eval_tokens=max_eval_tokens,
    )
    print(f"WikiText-2 {args.split} PPL: {ppl:.4f}")


if __name__ == "__main__":
    main()
