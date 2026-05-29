import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_bfp import add_bfp_to_llama
from llama_bfp_gptq import calibrate_and_apply_bfp_gptq_to_llama, save_bfp_gptq_weights
from llama_fuse import fuse_llama_model
from llama_rotation import rotate_llama_model


def resolve_optional_bits(explicit_bits, default_bits=None):
    if explicit_bits is None:
        return default_bits
    return explicit_bits if explicit_bits > 0 else None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rotation-block-size", type=int, default=32)
    parser.add_argument("--rotation-seed", type=int, default=0)
    parser.add_argument("--no-online-o-proj-had", action="store_true")
    parser.add_argument("--no-online-down-proj-had", action="store_true")
    parser.add_argument("--bfp", action="store_true")
    parser.add_argument("--v-bits", type=int, default=None)
    parser.add_argument("--v-groupsize", type=int, default=-1)
    parser.add_argument("--v-clip-ratio", type=float, default=1.0)
    parser.add_argument("--k-bits", type=int, default=None)
    parser.add_argument("--k-groupsize", type=int, default=-1)
    parser.add_argument("--k-clip-ratio", type=float, default=1.0)
    parser.add_argument("--no-qk-online-had", action="store_true")
    parser.add_argument("--bfp-gptq-block-size", type=int, default=32)
    parser.add_argument("--bfp-gptq-mantissa-bits", type=int, default=5)
    parser.add_argument("--bfp-gptq-lambda", type=float, default=1e-4)
    parser.add_argument("--bfp-gptq-no-weight-quant", action="store_true")
    parser.add_argument("--bfp-gptq-calib-samples", type=int, default=128)
    parser.add_argument("--bfp-gptq-calib-seqlen", type=int, default=2048)
    parser.add_argument("--bfp-gptq-calib-seed", type=int, default=0)
    parser.add_argument("--bfp-gptq-calib-split", default="train")
    parser.add_argument("--bfp-gptq-calib-mode", choices=["layerwise", "global"], default="layerwise")
    parser.add_argument("--no-bfp-gptq-forward-activations", action="store_true")
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
        "attn_implementation": args.attn_implementation,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    fuse_llama_model(model)
    rotate_llama_model(
        model,
        rotation_block_size=args.rotation_block_size,
        seed=args.rotation_seed,
        online_o_proj_had=not args.no_online_o_proj_had,
        online_down_proj_had=not args.no_online_down_proj_had,
    )

    default_kv_bits = 4 if args.bfp else None
    v_bits = resolve_optional_bits(args.v_bits, default_kv_bits)
    k_bits = resolve_optional_bits(args.k_bits, default_kv_bits)
    if args.bfp or v_bits is not None or k_bits is not None:
        counts = add_bfp_to_llama(
            model,
            a_bits=None,
            v_bits=v_bits,
            v_groupsize=args.v_groupsize,
            v_clip_ratio=args.v_clip_ratio,
            k_bits=k_bits,
            k_groupsize=args.k_groupsize,
            k_clip_ratio=args.k_clip_ratio,
            qk_online_had=not args.no_qk_online_had,
        )
        print(
            "BFP/QK calibration enabled: "
            f"activation=internal, v={counts['v']}, k={counts['k']}"
        )

    corrected = calibrate_and_apply_bfp_gptq_to_llama(
        model,
        tokenizer,
        calib_split=args.bfp_gptq_calib_split,
        calib_samples=args.bfp_gptq_calib_samples,
        calib_seqlen=args.bfp_gptq_calib_seqlen,
        calib_seed=args.bfp_gptq_calib_seed,
        block_size=args.bfp_gptq_block_size,
        mantissa_bits=args.bfp_gptq_mantissa_bits,
        lambda_reg=args.bfp_gptq_lambda,
        quantize_weight=not args.bfp_gptq_no_weight_quant,
        calib_mode=args.bfp_gptq_calib_mode,
        quantize_forward_activations=not args.no_bfp_gptq_forward_activations,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_bfp_gptq_weights(
        model,
        output,
        metadata={
            "model_name": args.model_name,
            "fused": True,
            "rotated": True,
            "rotation_block_size": args.rotation_block_size,
            "rotation_seed": args.rotation_seed,
            "runtime_bfp": args.bfp,
            "runtime_activation_bfp": not args.no_bfp_gptq_forward_activations,
            "runtime_v_bits": v_bits,
            "runtime_v_groupsize": args.v_groupsize,
            "runtime_v_clip_ratio": args.v_clip_ratio,
            "runtime_k_bits": k_bits,
            "runtime_k_groupsize": args.k_groupsize,
            "runtime_k_clip_ratio": args.k_clip_ratio,
            "runtime_qk_online_had": not args.no_qk_online_had,
            "bfp_gptq_block_size": args.bfp_gptq_block_size,
            "bfp_gptq_mantissa_bits": args.bfp_gptq_mantissa_bits,
            "bfp_gptq_lambda": args.bfp_gptq_lambda,
            "bfp_gptq_quantize_weight": not args.bfp_gptq_no_weight_quant,
            "bfp_gptq_calib_split": args.bfp_gptq_calib_split,
            "bfp_gptq_calib_mode": args.bfp_gptq_calib_mode,
            "bfp_gptq_calib_samples": args.bfp_gptq_calib_samples,
            "bfp_gptq_calib_seqlen": args.bfp_gptq_calib_seqlen,
            "bfp_gptq_calib_seed": args.bfp_gptq_calib_seed,
            "corrected_linears": corrected,
        },
    )
    print(f"Saved BFP-GPTQ corrected fused+rotated weights to {output}")


if __name__ == "__main__":
    main()
