import transformers
import argparse
from llama_fuse import fuse_llama_model
from llama_rotation import rotate_llama_model

def args_parser():
    pass

def main():
    args = args_parser()
    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model_name)
    model.eval()

    if args.rotate:
        fuse_llama_model(model)
        rotate_llama_model(
            model,
            rotation_block_size=getattr(args, "rotation_block_size", 32),
            seed=getattr(args, "rotation_seed", 0),
            online_o_proj_had=not getattr(args, "no_online_o_proj_had", False),
            online_down_proj_had=not getattr(args, "no_online_down_proj_had", False),
        )
