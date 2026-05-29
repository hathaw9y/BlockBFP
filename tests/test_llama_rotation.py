import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_rotation import (
    add_online_hadamard_to_linear,
    add_headwise_online_hadamard_to_linear,
    apply_headwise_hadamard_to_last_dim,
    apply_rotation_to_last_dim,
    build_rotation_matrix,
    resolve_rotation_block_size,
    rotate_weight_input,
    rotate_weight_output,
)


class LlamaRotationTest(unittest.TestCase):
    def test_default_block_size_is_block_random_hadamard(self):
        self.assertEqual(resolve_rotation_block_size(32, hidden_size=4096), 32)

    def test_rotation_matrix_is_orthogonal(self):
        q = build_rotation_matrix(8, block_size=2, seed=123)

        self.assertTrue(torch.allclose(q @ q.t(), torch.eye(8), atol=1e-6))

    def test_structured_rotation_matches_dense_matrix(self):
        torch.manual_seed(0)
        x = torch.randn(3, 8)
        q = build_rotation_matrix(8, block_size=2, seed=123)

        self.assertTrue(
            torch.allclose(apply_rotation_to_last_dim(x, block_size=2, seed=123), x @ q)
        )

    def test_input_weight_rotation_preserves_linear_projection(self):
        torch.manual_seed(0)
        linear = torch.nn.Linear(4, 3, bias=False)
        x = torch.randn(2, 4)
        q = build_rotation_matrix(4, block_size=2, seed=0)

        expected = linear(x)
        rotate_weight_input(linear, q)
        actual = linear(x @ q)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_output_weight_rotation_moves_output_to_rotated_basis(self):
        torch.manual_seed(0)
        linear = torch.nn.Linear(5, 4)
        x = torch.randn(2, 5)
        q = build_rotation_matrix(4, block_size=2, seed=0)

        expected = linear(x) @ q
        rotate_weight_output(linear, q)
        actual = linear(x)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_online_hadamard_pair_preserves_linear_projection(self):
        torch.manual_seed(0)
        linear = torch.nn.Linear(8, 3, bias=False)
        x = torch.randn(2, 8)
        expected = linear(x)

        add_online_hadamard_to_linear(linear, block_size=2)
        actual = linear(x)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_headwise_online_hadamard_does_not_mix_heads(self):
        x = torch.zeros(1, 12)
        x[:, :4] = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        actual = apply_headwise_hadamard_to_last_dim(
            x,
            num_heads=3,
            head_dim=4,
            block_size=2,
        )

        self.assertTrue(torch.any(actual[:, :4] != 0))
        self.assertTrue(torch.allclose(actual[:, 4:], torch.zeros(1, 8)))

    def test_headwise_online_hadamard_pair_preserves_o_proj_projection(self):
        torch.manual_seed(0)
        linear = torch.nn.Linear(12, 5, bias=False)
        x = torch.randn(2, 12)
        expected = linear(x)

        add_headwise_online_hadamard_to_linear(
            linear,
            num_heads=3,
            head_dim=4,
            block_size=2,
        )
        actual = linear(x)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
