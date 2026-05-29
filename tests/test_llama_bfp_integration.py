import sys
import unittest
from pathlib import Path

import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_bfp import add_bfp_to_llama
from llama_fuse import fuse_llama_model
from llama_rotation import rotate_llama_model


class LlamaBFPIntegrationTest(unittest.TestCase):
    def test_tiny_llama_forward_after_fuse_rotate_bfp(self):
        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
        model = LlamaForCausalLM(config)
        model.eval()

        fuse_llama_model(model)
        rotate_llama_model(model, rotation_block_size=8, seed=0)
        counts = add_bfp_to_llama(
            model,
            a_bits=4,
            a_groupsize=8,
            v_bits=4,
            v_groupsize=8,
            k_bits=4,
            k_groupsize=8,
        )

        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            outputs = model(input_ids)

        self.assertEqual(outputs.logits.shape, (1, 8, config.vocab_size))
        self.assertGreater(counts["activation"], 0)
        self.assertEqual(counts["v"], config.num_hidden_layers)
        self.assertEqual(counts["k"], config.num_hidden_layers)


if __name__ == "__main__":
    unittest.main()
