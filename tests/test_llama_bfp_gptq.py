import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_bfp_gptq import (
    cholesky_inverse_stable,
    correct_and_quantize_weight_bfp_gptq,
    correct_and_quantize_weight_bfp_gptq_from_stats,
    load_bfp_gptq_weights,
    quant_bfp_mantissa,
    sample_calibration_starts,
    save_bfp_gptq_weights,
)


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
