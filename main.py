import transformers
import argparse
from llama_fuse import fuse_llama_model

def args_parser():
    pass

def main():
    args = args_parser()
    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model_name)
    model.eval()

    if args.rotate:
        fuse_llama_model(model)
        rotation_utils.rotate_model(model, args)
