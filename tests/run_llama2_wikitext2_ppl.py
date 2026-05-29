import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_bfp import add_bfp_to_llama
from llama_bfp_gptq import calibrate_and_apply_bfp_gptq_to_llama, load_bfp_gptq_weights
from llama_fuse import fuse_llama_model
from llama_rotation import rotate_llama_model
from ppl_eval import evaluate_wikitext2_ppl


def resolve_optional_bits(explicit_bits, default_bits=None):
    if explicit_bits is None:
        return default_bits
    return explicit_bits if explicit_bits > 0 else None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--attn-implementation", default="eager")
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
    parser.add_argument("--bfp", action="store_true")
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--a-groupsize", type=int, default=-1)
    parser.add_argument("--a-clip-ratio", type=float, default=1.0)
    parser.add_argument("--no-a-bfp", action="store_true")
    parser.add_argument("--v-bits", type=int, default=None)
    parser.add_argument("--v-groupsize", type=int, default=-1)
    parser.add_argument("--v-clip-ratio", type=float, default=1.0)
    parser.add_argument("--k-bits", type=int, default=None)
    parser.add_argument("--k-groupsize", type=int, default=-1)
    parser.add_argument("--k-clip-ratio", type=float, default=1.0)
    parser.add_argument("--no-qk-online-had", action="store_true")
    parser.add_argument("--qk-online-had-only", action="store_true")
    parser.add_argument("--bfp-gptq", action="store_true")
    parser.add_argument("--bfp-gptq-load", default=None)
    parser.add_argument("--bfp-gptq-block-size", type=int, default=32)
    parser.add_argument("--bfp-gptq-mantissa-bits", type=int, default=5)
    parser.add_argument("--bfp-gptq-lambda", type=float, default=1e-4)
    parser.add_argument("--bfp-gptq-no-weight-quant", action="store_true")
    parser.add_argument("--bfp-gptq-calib-samples", type=int, default=128)
    parser.add_argument("--bfp-gptq-calib-seqlen", type=int, default=2048)
    parser.add_argument("--bfp-gptq-calib-seed", type=int, default=0)
    parser.add_argument("--bfp-gptq-calib-split", default="train")
    parser.add_argument("--bfp-gptq-calib-mode", choices=["layerwise", "global"], default="layerwise")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bfp_gptq and not args.rotate:
        raise ValueError("--bfp-gptq requires --rotate so W is the Hadamard-rotated weight.")
    if args.bfp_gptq_load is not None and not args.rotate:
        raise ValueError("--bfp-gptq-load requires --rotate so the saved rotated weights match.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )

    model_kwargs = {
        "local_files_only": args.local_files_only,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": args.attn_implementation,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    quant_or_qk_enabled = (
        args.bfp
        or args.v_bits is not None
        or args.k_bits is not None
        or args.qk_online_had_only
        or args.bfp_gptq
        or args.bfp_gptq_load is not None
    )
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
    if args.bfp_gptq:
        corrected = calibrate_and_apply_bfp_gptq_to_llama(
            model,
            tokenizer,
            calib_split=args.bfp_gptq_calib_split,
            calib_samples=args.bfp_gptq_calib_samples,
            calib_seqlen=args.bfp_gptq_calib_seqlen,
            calib_seed=args.bfp_gptq_calib_seed,
            block_size=args.bfp_gptq_block_size,
            mantissa_bits=args.bfp_gptq_mantissa_bits,
            activation_bits=args.a_bits,
            activation_clip_ratio=args.a_clip_ratio,
            lambda_reg=args.bfp_gptq_lambda,
            quantize_weight=not args.bfp_gptq_no_weight_quant,
            calib_mode=args.bfp_gptq_calib_mode,
        )
        mode = "correction-only fp weights" if args.bfp_gptq_no_weight_quant else "corrected+bfp weights"
        print(f"BFP-GPTQ corrected linears on fused+rotated weights ({mode}): {corrected}")
    if args.bfp_gptq_load is not None:
        loaded = load_bfp_gptq_weights(model, args.bfp_gptq_load)
        print(f"BFP-GPTQ loaded corrected fused+rotated linears: {loaded}")
    if quant_or_qk_enabled:
        activation_bits = args.a_bits if args.bfp and not args.no_a_bfp else None
        default_kv_bits = 4 if args.bfp else None
        v_bits = resolve_optional_bits(args.v_bits, default_kv_bits)
        k_bits = resolve_optional_bits(args.k_bits, default_kv_bits)
        counts = add_bfp_to_llama(
            model,
            a_bits=activation_bits,
            a_groupsize=args.a_groupsize,
            a_clip_ratio=args.a_clip_ratio,
            v_bits=v_bits,
            v_groupsize=args.v_groupsize,
            v_clip_ratio=args.v_clip_ratio,
            k_bits=k_bits,
            k_groupsize=args.k_groupsize,
            k_clip_ratio=args.k_clip_ratio,
            qk_online_had=not args.no_qk_online_had,
            force_qk_online_had=args.qk_online_had_only,
        )
        print(
            "BFP/QK enabled: "
            f"activation={counts['activation']}, v={counts['v']}, k={counts['k']}"
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
    if args.qk_online_had_only and not args.bfp and args.v_bits is None and args.k_bits is None:
        fuse_label = "rotated+qk-had" if args.rotate else "qk-had"
    elif args.bfp_gptq and not args.bfp and args.v_bits is None and args.k_bits is None:
        fuse_label = "rotated+bfp-gptq" if args.rotate else "bfp-gptq"
    elif args.bfp_gptq_load is not None and not args.bfp and args.v_bits is None and args.k_bits is None:
        fuse_label = "rotated+bfp-gptq-loaded"
    elif quant_or_qk_enabled:
        fuse_label = "rotated+bfp" if args.rotate else "bfp"
    elif args.rotate:
        fuse_label = "fused+rotated"
    elif args.fuse:
        fuse_label = "fused"
    else:
        fuse_label = "baseline"
    print(f"WikiText-2 {args.split} PPL ({fuse_label}): {ppl:.4f}")


if __name__ == "__main__":
    main()
