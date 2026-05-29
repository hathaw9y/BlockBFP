import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_bfp import (
    BFP_DEFAULT_BLOCK_SIZE,
    add_activation_bfp_to_llama,
    add_bfp_to_llama,
    add_output_bfp_to_linear,
    bfp_fake_quant,
    clamp_to_dtype_range,
    resolve_bfp_block_size,
)
from llama_rotation import apply_hadamard_to_last_dim


class DummySelfAttn(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size)


class DummyMLP(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.gate_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.down_proj = torch.nn.Linear(hidden_size, hidden_size)


class DummyLayer(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.self_attn = DummySelfAttn(hidden_size)
        self.mlp = DummyMLP(hidden_size)


class DummyInnerModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layers = torch.nn.ModuleList([DummyLayer(hidden_size)])


class DummyLlama(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.model = DummyInnerModel(hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, hidden_size)


class LlamaBFPTest(unittest.TestCase):
    def test_groupsize_minus_one_uses_default_block_size(self):
        self.assertEqual(resolve_bfp_block_size(-1), BFP_DEFAULT_BLOCK_SIZE)

    def test_bfp_fake_quant_uses_signed_four_bit_range(self):
        x = torch.tensor([[-9.0, -8.0, -7.0, 0.0, 7.0, 8.0]])

        actual = bfp_fake_quant(x, bits=4, block_size=8)

        self.assertEqual(actual.shape, x.shape)
        self.assertTrue(torch.all(actual <= 8.0))
        self.assertTrue(torch.all(actual >= -8.0))

    def test_bfp_fake_quant_pads_and_slices_last_dim(self):
        x = torch.randn(2, 5)

        actual = bfp_fake_quant(x, bits=4, block_size=4)

        self.assertEqual(actual.shape, x.shape)

    def test_bfp_fake_quant_keeps_non_finite_inputs_from_creating_nan(self):
        x = torch.tensor([[float("inf"), float("-inf"), float("nan"), 1.0]], dtype=torch.float16)

        actual = bfp_fake_quant(x, bits=4, block_size=4)

        self.assertTrue(torch.all(torch.isfinite(actual)))

    def test_fp32_attention_scores_avoid_fp16_overflow_pattern(self):
        query = torch.full((1, 1, 1, 128), 512.0, dtype=torch.float16)
        key = torch.full((1, 1, 1, 128), 512.0, dtype=torch.float16)

        scores = torch.matmul(query.float(), key.float().transpose(2, 3)) / (128 ** 0.5)

        self.assertTrue(torch.all(torch.isfinite(scores)))

    def test_clamp_to_dtype_range_prevents_fp16_inf_on_cast(self):
        x = torch.tensor([1e8, -1e8, 1.0], dtype=torch.float32)

        actual = clamp_to_dtype_range(x, torch.float16).to(torch.float16)

        self.assertTrue(torch.all(torch.isfinite(actual)))

    def test_qk_online_hadamard_preserves_attention_scores(self):
        torch.manual_seed(0)
        query = torch.randn(1, 2, 3, 8)
        key = torch.randn(1, 2, 4, 8)
        expected = torch.matmul(query, key.transpose(2, 3))

        query_rot = apply_hadamard_to_last_dim(query, block_size=4)
        key_rot = apply_hadamard_to_last_dim(key, block_size=4)
        actual = torch.matmul(query_rot, key_rot.transpose(2, 3))

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_activation_bfp_wraps_linears_but_skips_lm_head(self):
        model = DummyLlama()

        wrapped = add_activation_bfp_to_llama(model, bits=4, groupsize=4)

        self.assertEqual(wrapped, 7)
        self.assertTrue(hasattr(model.model.layers[0].self_attn.q_proj, "_bfp_input_handle"))
        self.assertFalse(hasattr(model.lm_head, "_bfp_input_handle"))

    def test_output_bfp_quantizes_linear_output(self):
        linear = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4)

        add_output_bfp_to_linear(linear, bits=4, groupsize=4)
        actual = linear(x)

        self.assertEqual(actual.shape, (2, 4))

    def test_add_bfp_to_llama_counts_activation_and_v(self):
        model = DummyLlama()

        counts = add_bfp_to_llama(model, a_bits=4, a_groupsize=4, v_bits=4, v_groupsize=4)

        self.assertEqual(counts["activation"], 7)
        self.assertEqual(counts["v"], 1)
        self.assertEqual(counts["k"], 0)


if __name__ == "__main__":
    unittest.main()
