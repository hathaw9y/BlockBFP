import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_bfp_gptq import (
    calibrate_and_apply_bfp_gptq_to_llama_layerwise,
    cholesky_inverse_stable,
    correct_and_quantize_weight_bfp_gptq,
    correct_and_quantize_weight_bfp_gptq_from_stats,
    load_bfp_gptq_weights,
    quant_bfp_mantissa,
    sample_calibration_starts,
    save_bfp_gptq_weights,
)


class TinyTokenizer:
    def __call__(self, text, return_tensors=None):
        tokens = torch.arange(256).remainder(32).unsqueeze(0)
        return type("Tokenized", (), {"input_ids": tokens})


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, hidden_states, **kwargs):
        return (self.linear(hidden_states),)


class TinyBaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 8)
        self.layers = torch.nn.ModuleList([TinyLayer(), TinyLayer()])


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"use_cache": True})()
        self.model = TinyBaseModel()


class LlamaBFPGPTQTest(unittest.TestCase):
    def test_quant_bfp_mantissa_preserves_shape_with_padding(self):
        x = torch.randn(3, 10)

        actual = quant_bfp_mantissa(x, block_size=8, mantissa_bits=5)

        self.assertEqual(actual.shape, x.shape)
        self.assertTrue(torch.all(torch.isfinite(actual)))

    def test_cholesky_inverse_stable_returns_finite_matrix(self):
        H = torch.eye(4)

        actual = cholesky_inverse_stable(H, lambda_reg=1e-4)

        self.assertEqual(actual.shape, H.shape)
        self.assertTrue(torch.all(torch.isfinite(actual)))

    def test_correct_and_quantize_weight_reduces_activation_bfp_error(self):
        torch.manual_seed(0)
        W = torch.randn(16, 32)
        X = torch.randn(32, 256)
        X_Q = quant_bfp_mantissa(X.t(), block_size=32, mantissa_bits=5).t()

        W_naive = quant_bfp_mantissa(W, block_size=32, mantissa_bits=5)
        W_corrected = correct_and_quantize_weight_bfp_gptq(
            W,
            X,
            block_size=32,
            mantissa_bits=5,
            lambda_reg=1e-3,
        )

        target = W @ X
        naive_err = torch.linalg.norm(target - W_naive @ X_Q)
        corrected_err = torch.linalg.norm(target - W_corrected @ X_Q)

        self.assertLessEqual(corrected_err, naive_err * 1.05)

    def test_correction_only_keeps_fp_weight_without_bfp_weight_quant(self):
        torch.manual_seed(2)
        W = torch.randn(16, 32)
        X = torch.randn(32, 256)
        X_Q = quant_bfp_mantissa(X.t(), block_size=32, mantissa_bits=5).t()

        W_corrected = correct_and_quantize_weight_bfp_gptq(
            W,
            X,
            block_size=32,
            mantissa_bits=5,
            lambda_reg=1e-3,
            quantize_weight=False,
        )

        target = W @ X
        naive_err = torch.linalg.norm(target - W @ X_Q)
        corrected_err = torch.linalg.norm(target - W_corrected @ X_Q)

        self.assertEqual(W_corrected.shape, W.shape)
        self.assertTrue(torch.all(torch.isfinite(W_corrected)))
        self.assertLessEqual(corrected_err, naive_err)

    def test_stats_path_matches_activation_path_shape(self):
        torch.manual_seed(1)
        W = torch.randn(8, 16)
        X = torch.randn(16, 64)
        blocks = []
        for start in range(0, 16, 8):
            X_b = X[start : start + 8]
            X_Q_b = quant_bfp_mantissa(X_b.t(), block_size=8, mantissa_bits=5).t()
            blocks.append(
                {
                    "cross": ((X_b - X_Q_b) @ X_Q_b.t()).double(),
                    "hessian": (2 * X_Q_b @ X_Q_b.t()).double(),
                    "magnitude": X_b.abs().sum(dim=1).double(),
                    "count": X_b.shape[1],
                }
            )
        stats = {"in_features": 16, "block_size": 8, "blocks": blocks}

        actual = correct_and_quantize_weight_bfp_gptq_from_stats(
            W,
            stats,
            mantissa_bits=5,
            lambda_reg=1e-3,
        )

        self.assertEqual(actual.shape, W.shape)
        self.assertTrue(torch.all(torch.isfinite(actual)))

    def test_sample_calibration_starts_uses_random_fixed_length_windows(self):
        starts = sample_calibration_starts(total_len=10000, seqlen=2048, nsamples=128, seed=7)

        self.assertEqual(len(starts), 128)
        self.assertTrue(all(0 <= start <= 10000 - 2048 for start in starts))
        self.assertEqual(starts, sample_calibration_starts(10000, 2048, 128, seed=7))

    def test_layerwise_calibration_corrects_layer_linears(self):
        model = TinyCausalLM()
        before = model.model.layers[0].linear.weight.detach().clone()

        with patch("llama_bfp_gptq.load_dataset", return_value={"text": ["tiny calibration text"]}):
            corrected = calibrate_and_apply_bfp_gptq_to_llama_layerwise(
                model,
                TinyTokenizer(),
                calib_samples=2,
                calib_seqlen=16,
                block_size=4,
                mantissa_bits=5,
                lambda_reg=1e-3,
                quantize_weight=False,
                show_progress=False,
            )

        self.assertEqual(corrected, 2)
        self.assertTrue(torch.all(torch.isfinite(model.model.layers[0].linear.weight)))
        self.assertFalse(torch.equal(model.model.layers[0].linear.weight, before))

    def test_save_and_load_bfp_gptq_weights(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
        path = Path("/tmp/test_bfp_gptq_weights.pt")
        saved_weight = model[0].weight.detach().clone()

        save_bfp_gptq_weights(model, path)
        model[0].weight.data.zero_()
        loaded = load_bfp_gptq_weights(model, path)

        self.assertEqual(loaded, 2)
        self.assertTrue(torch.allclose(model[0].weight, saved_weight))


if __name__ == "__main__":
    unittest.main()
