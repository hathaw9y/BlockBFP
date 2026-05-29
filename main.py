import transformers
import argparse
from llama_bfp import add_bfp_to_llama
from llama_fuse import fuse_llama_model
from llama_rotation import rotate_llama_model

def args_parser():
    pass

def main():
    args = args_parser()
    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model_name)
    model.eval()

    bfp_enabled = getattr(args, "bfp", False)
    if args.rotate:
        fuse_llama_model(model)
        rotate_llama_model(
            model,
            rotation_block_size=getattr(args, "rotation_block_size", 32),
            seed=getattr(args, "rotation_seed", 0),
            online_o_proj_had=not getattr(args, "no_online_o_proj_had", False),
            online_down_proj_had=not getattr(args, "no_online_down_proj_had", False),
        )
    if bfp_enabled:
        add_bfp_to_llama(
            model,
            a_bits=None if getattr(args, "no_a_bfp", False) else getattr(args, "a_bits", 4),
            a_groupsize=getattr(args, "a_groupsize", -1),
            a_clip_ratio=getattr(args, "a_clip_ratio", 1.0),
            v_bits=getattr(args, "v_bits", None),
            v_groupsize=getattr(args, "v_groupsize", -1),
            v_clip_ratio=getattr(args, "v_clip_ratio", 1.0),
            k_bits=getattr(args, "k_bits", None),
            k_groupsize=getattr(args, "k_groupsize", -1),
            k_clip_ratio=getattr(args, "k_clip_ratio", 1.0),
            qk_online_had=not getattr(args, "no_qk_online_had", False),
            force_qk_online_had=getattr(args, "qk_online_had_only", False),
        )
