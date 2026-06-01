import unittest

import torch

from src.model_utils import ZeroAttention


class ZeroAttentionTest(unittest.TestCase):
    def test_returns_current_two_value_attention_contract(self) -> None:
        hidden_states = torch.randn(2, 3, 4)
        outputs = ZeroAttention(layer_idx=7)(hidden_states, use_cache=False)

        self.assertEqual(len(outputs), 2)
        self.assertTrue(torch.equal(outputs[0], torch.zeros_like(hidden_states)))
        self.assertIsNone(outputs[1])


if __name__ == "__main__":
    unittest.main()
