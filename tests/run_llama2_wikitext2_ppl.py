import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_fuse import fuse_llama_model
from llama_rotation import rotate_llama_model
from ppl_eval import evaluate_wikitext2_ppl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--max-eval-tokens", type=int, default=0)
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rotation-block-size", type=int, default=32)
    parser.add_argument("--rotation-seed", type=int, default=0)
    parser.add_argument("--no-online-o-proj-had", action="store_true")
    parser.add_argument("--no-online-down-proj-had", action="store_true")
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
    if args.fuse or args.rotate:
        fuse_llama_model(model)
    if args.rotate:
        rotate_llama_model(
            model,
            rotation_block_size=args.rotation_block_size,
            seed=args.rotation_seed,
            online_o_proj_had=not args.no_online_o_proj_had,
            online_down_proj_had=not args.no_online_down_proj_had,
        )

    max_eval_tokens = None if args.max_eval_tokens <= 0 else args.max_eval_tokens
    ppl = evaluate_wikitext2_ppl(
        model,
        tokenizer,
        split=args.split,
        max_length=args.max_length,
        stride=args.stride,
        max_eval_tokens=max_eval_tokens,
    )
    if args.rotate:
        fuse_label = "fused+rotated"
    elif args.fuse:
        fuse_label = "fused"
    else:
        fuse_label = "baseline"
    print(f"WikiText-2 {args.split} PPL ({fuse_label}): {ppl:.4f}")


if __name__ == "__main__":
    main()
