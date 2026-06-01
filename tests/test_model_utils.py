import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.model_utils import ZeroAttention, ZeroMLP, drop_layers


class DecoderLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Identity()
        self.mlp = nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class ModelLike:
    def __init__(self) -> None:
        self.config = SimpleNamespace(model_type="mistral")
        self.model = SimpleNamespace(layers=nn.ModuleList([DecoderLike()]))


class ZeroAttentionTest(unittest.TestCase):
    def test_returns_current_two_value_attention_contract(self) -> None:
        hidden_states = torch.randn(2, 3, 4)
        outputs = ZeroAttention(layer_idx=7)(hidden_states, use_cache=False)

        self.assertEqual(len(outputs), 2)
        self.assertTrue(torch.equal(outputs[0], torch.zeros_like(hidden_states)))
        self.assertIsNone(outputs[1])

    def test_attn_and_mlp_drop_keeps_decoder_tensor_contract(self) -> None:
        model = ModelLike()
        original_layer = model.model.layers[0]
        hidden_states = torch.randn(2, 3, 4)

        drop_layers(model, ["attn+mlp"])
        outputs = model.model.layers[0](hidden_states)

        self.assertIs(model.model.layers[0], original_layer)
        self.assertIsInstance(model.model.layers[0].self_attn, ZeroAttention)
        self.assertIsInstance(model.model.layers[0].mlp, ZeroMLP)
        self.assertTrue(torch.equal(outputs, hidden_states))


if __name__ == "__main__":
    unittest.main()
