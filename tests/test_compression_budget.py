import random
import tempfile
import unittest
from pathlib import Path

import torch

from src.compression_budget import (
    candidate_compression_cost,
    flatten_quant_state,
    repair_quant_state_to_budget,
    uniform_quantization_target_cost,
    validate_exact_budget,
)


class Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.o_proj = torch.nn.Linear(4, 4, bias=False)


class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(4, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 4, bias=False)


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attention()
        self.mlp = MLP()
        self.input_norm = torch.nn.LayerNorm(4)


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.layers = torch.nn.ModuleList([Block(), Block()])
        self.lm_head = torch.nn.Linear(4, 8, bias=False)


class WeightOnly(torch.nn.Module):
    def __init__(self, out_features: int, in_features: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device="meta")
        )


class MistralAttentionShape(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = WeightOnly(4096, 4096)
        self.k_proj = WeightOnly(1024, 4096)
        self.v_proj = WeightOnly(1024, 4096)
        self.o_proj = WeightOnly(4096, 4096)


class MistralMLPShape(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = WeightOnly(14336, 4096)
        self.up_proj = WeightOnly(14336, 4096)
        self.down_proj = WeightOnly(4096, 14336)


class MistralBlockShape(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = MistralAttentionShape()
        self.mlp = MistralMLPShape()
        self.input_layernorm = torch.nn.Parameter(torch.empty(4096, device="meta"))
        self.post_attention_layernorm = torch.nn.Parameter(
            torch.empty(4096, device="meta")
        )


class MistralBodyShape(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = WeightOnly(32768, 4096)
        self.layers = torch.nn.ModuleList([MistralBlockShape() for _ in range(32)])
        self.norm = torch.nn.Parameter(torch.empty(4096, device="meta"))


class Mistral7BV03Shape(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = MistralBodyShape()
        self.lm_head = WeightOnly(32768, 4096)


class CompressionBudgetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeModel().half()
        self.attention_names = [
            "layers.0.self_attn",
            "layers.1.self_attn",
        ]
        self.mlp_names = ["layers.0.mlp", "layers.1.mlp"]
        self.module_names = [
            "layers.0.self_attn.q_proj",
            "layers.0.self_attn.o_proj",
            "layers.1.self_attn.q_proj",
            "layers.1.self_attn.o_proj",
            "layers.0.mlp.up_proj",
            "layers.0.mlp.down_proj",
            "layers.1.mlp.up_proj",
            "layers.1.mlp.down_proj",
        ]
        self.groups = [self.module_names]
        self.cost_kwargs = {
            "attention_module_names": self.attention_names,
            "mlp_module_names": self.mlp_names,
            "dense_dtype_bits": 16,
            "group_size": 4,
            "include_quantization_metadata": True,
            "scale_bits": 16,
            "zero_point_bits": 16,
        }

    def uniform_state(self, bitwidth: int = 3):
        return [[bitwidth] * len(self.module_names)]

    def joint_candidate(self, drop_state, bitwidth: int = 3):
        return {"drop": drop_state, "quant": self.uniform_state(bitwidth)}

    def test_quant_only_uniform_cost_equals_configured_target(self) -> None:
        target = uniform_quantization_target_cost(
            self.model,
            self.groups,
            3,
            **self.cost_kwargs,
        )
        realized = candidate_compression_cost(
            self.model,
            self.uniform_state(3),
            grouped_layer_names=self.groups,
            **self.cost_kwargs,
        )
        self.assertEqual(realized, target)
        self.assertTrue(validate_exact_budget(realized, target["total_cost_bits"]))

    def test_removed_modules_and_assignments_cost_zero(self) -> None:
        drop = {"attn": [True, False], "mlp": [False, False]}
        candidate = self.joint_candidate(drop)
        base = candidate_compression_cost(
            self.model,
            candidate,
            grouped_layer_names=self.groups,
            **self.cost_kwargs,
        )
        candidate["quant"][0][0] = 6
        candidate["quant"][0][1] = 2
        changed_dropped_assignments = candidate_compression_cost(
            self.model,
            candidate,
            grouped_layer_names=self.groups,
            **self.cost_kwargs,
        )
        self.assertEqual(
            changed_dropped_assignments["total_cost_bits"],
            base["total_cost_bits"],
        )
        self.assertEqual(base["removed_parameters"], 32)
        self.assertEqual(base["searched_parameters_active"], 160)

    def test_increasing_active_bitwidth_increases_exact_weight_cost(self) -> None:
        state = self.uniform_state(3)
        base = candidate_compression_cost(
            self.model,
            state,
            grouped_layer_names=self.groups,
            **self.cost_kwargs,
        )
        state[0][0] = 4
        increased = candidate_compression_cost(
            self.model,
            state,
            grouped_layer_names=self.groups,
            **self.cost_kwargs,
        )
        self.assertEqual(increased["total_cost_bits"] - base["total_cost_bits"], 16)
        self.assertEqual(
            increased["quantization_metadata_bits"],
            base["quantization_metadata_bits"],
        )

    def test_budget_neutral_depth_precision_trade_is_repaired_exactly(self) -> None:
        target = uniform_quantization_target_cost(
            self.model,
            self.groups,
            3,
            **self.cost_kwargs,
        )
        drop = {"attn": [True, False], "mlp": [False, False]}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for module_name in self.module_names:
                module_dir = root / module_name
                module_dir.mkdir(parents=True)
                for bitwidth in range(2, 7):
                    (module_dir / f"{bitwidth}.pth").touch()
            repaired = repair_quant_state_to_budget(
                self.model,
                self.groups,
                root,
                self.uniform_state(3),
                drop,
                target["total_cost_bits"],
                attention_module_names=self.attention_names,
                mlp_module_names=self.mlp_names,
                dense_dtype_bits=16,
                group_size=4,
                include_quantization_metadata=True,
                scale_bits=16,
                zero_point_bits=16,
                rng=random.Random(0),
            )
        realized = candidate_compression_cost(
            self.model,
            {"drop": drop, "quant": repaired},
            grouped_layer_names=self.groups,
            **self.cost_kwargs,
        )
        self.assertTrue(validate_exact_budget(realized, target["total_cost_bits"]))
        active_assignments = flatten_quant_state(self.groups, repaired)
        self.assertGreater(
            sum(
                self.model.get_submodule(name).weight.numel() * level
                for name, level in active_assignments.items()
                if not name.startswith("layers.0.self_attn.")
            )
            / 160,
            3.0,
        )

    def test_mistral_exact_reference_and_25_percent_trade(self) -> None:
        model = Mistral7BV03Shape()
        attention_names = [f"model.layers.{index}.self_attn" for index in range(32)]
        mlp_names = [f"model.layers.{index}.mlp" for index in range(32)]
        module_names = [
            f"model.layers.{index}.{component}.{projection}"
            for index in range(32)
            for component, projections in (
                ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                ("mlp", ("gate_proj", "up_proj", "down_proj")),
            )
            for projection in projections
        ]
        groups_by_size = {}
        for name in module_names:
            groups_by_size.setdefault(model.get_submodule(name).weight.numel(), []).append(name)
        groups = [groups_by_size[size] for size in sorted(groups_by_size)]
        cost_kwargs = {
            "attention_module_names": attention_names,
            "mlp_module_names": mlp_names,
            "dense_dtype_bits": 16,
            "group_size": 128,
            "include_quantization_metadata": True,
            "scale_bits": 16,
            "zero_point_bits": 16,
        }
        target = uniform_quantization_target_cost(model, groups, 3, **cost_kwargs)
        self.assertEqual(target["total_parameters_dense"], 7_248_023_552)
        self.assertEqual(target["searched_parameters_dense"], 6_979_321_856)
        self.assertEqual(target["dense_model_bits"], 115_968_376_832)
        self.assertEqual(target["quantization_metadata_bits"], 1_744_830_464)
        self.assertEqual(target["total_cost_bits"], 26_982_023_168)

        drop = {
            "attn": [index < 8 for index in range(32)],
            "mlp": [8 <= index < 16 for index in range(32)],
        }
        state = [[3] * len(group) for group in groups]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in module_names:
                module_dir = root / name
                module_dir.mkdir(parents=True)
                for bitwidth in range(2, 7):
                    (module_dir / f"{bitwidth}.pth").touch()
            repaired = repair_quant_state_to_budget(
                model,
                groups,
                root,
                state,
                drop,
                target["total_cost_bits"],
                attention_module_names=attention_names,
                mlp_module_names=mlp_names,
                dense_dtype_bits=16,
                group_size=128,
                include_quantization_metadata=True,
                scale_bits=16,
                zero_point_bits=16,
                preserve_equal_size_group_costs=True,
                uniform_reference_bitwidth=3,
                rng=random.Random(0),
            )
        realized = candidate_compression_cost(
            model,
            {"drop": drop, "quant": repaired},
            grouped_layer_names=groups,
            **cost_kwargs,
        )
        self.assertEqual(realized["total_cost_bits"], target["total_cost_bits"])
        self.assertAlmostEqual(realized["assigned_average_bitwidth_active"], 49 / 12)
        active_level_sums = []
        for group, levels in zip(groups, repaired):
            active_level_sums.append(
                sum(
                    level
                    for name, level in zip(group, levels)
                    if not (
                        ".self_attn." in name and drop["attn"][int(name.split(".")[2])]
                    )
                    and not (
                        ".mlp." in name and drop["mlp"][int(name.split(".")[2])]
                    )
                )
            )
        self.assertEqual(active_level_sums, [196, 196, 294])


if __name__ == "__main__":
    unittest.main()
