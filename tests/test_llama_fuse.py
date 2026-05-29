import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_fuse import RMSN, fuse_llama_model


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps


class DummySelfAttn(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)


class DummyMLP(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)


class DummyLayer(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size)
        self.self_attn = DummySelfAttn(hidden_size)
        self.mlp = DummyMLP(hidden_size)


class DummyInnerModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(3, hidden_size)
        self.layers = torch.nn.ModuleList([DummyLayer(hidden_size)])
        self.norm = LlamaRMSNorm(hidden_size)


class DummyConfig:
    hidden_size = 4


class DummyLlamaForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()
        self.model = DummyInnerModel(self.config.hidden_size)
        self.lm_head = torch.nn.Linear(self.config.hidden_size, 2, bias=False)


def fill_weight(module, start):
    values = torch.arange(start, start + module.weight.numel(), dtype=torch.float32)
    module.weight.data.copy_(values.reshape_as(module.weight))


class LlamaFuseTest(unittest.TestCase):
    def test_fuse_llama_model_targets_expected_layers(self):
        model = DummyLlamaForCausalLM()
        layer = model.model.layers[0]

        model.model.embed_tokens.weight.data.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, 4.0, 6.0, 8.0],
                    [-1.0, 0.0, 1.0, 2.0],
                ]
            )
        )
        layer.input_layernorm.weight.data.copy_(torch.tensor([2.0, 3.0, 4.0, 5.0]))
        layer.post_attention_layernorm.weight.data.copy_(
            torch.tensor([7.0, 11.0, 13.0, 17.0])
        )
        model.model.norm.weight.data.copy_(torch.tensor([19.0, 23.0, 29.0, 31.0]))

        linear_modules = [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.up_proj,
            layer.mlp.gate_proj,
            layer.mlp.down_proj,
            model.lm_head,
        ]
        for idx, linear in enumerate(linear_modules):
            fill_weight(linear, idx * 100)

        saved = {id(linear): linear.weight.detach().clone() for linear in linear_modules}

        fuse_llama_model(model)

        self.assertTrue(
            torch.allclose(
                model.model.embed_tokens.weight.mean(dim=-1),
                torch.zeros(3),
            )
        )

        for linear in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
            self.assertTrue(
                torch.allclose(
                    linear.weight,
                    saved[id(linear)] * torch.tensor([2.0, 3.0, 4.0, 5.0]),
                )
            )
        self.assertTrue(torch.allclose(layer.self_attn.o_proj.weight, saved[id(layer.self_attn.o_proj)]))

        for linear in [layer.mlp.up_proj, layer.mlp.gate_proj]:
            self.assertTrue(
                torch.allclose(
                    linear.weight,
                    saved[id(linear)] * torch.tensor([7.0, 11.0, 13.0, 17.0]),
                )
            )
        self.assertTrue(torch.allclose(layer.mlp.down_proj.weight, saved[id(layer.mlp.down_proj)]))

        self.assertTrue(
            torch.allclose(
                model.lm_head.weight,
                saved[id(model.lm_head)] * torch.tensor([19.0, 23.0, 29.0, 31.0]),
            )
        )
        self.assertIsInstance(layer.input_layernorm, RMSN)
        self.assertIsInstance(layer.post_attention_layernorm, RMSN)
        self.assertIsInstance(model.model.norm, RMSN)


if __name__ == "__main__":
    unittest.main()
