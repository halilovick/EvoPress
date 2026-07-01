import random
import tempfile
import unittest
from pathlib import Path

from evo_joint_search import (
    adaptive_mutation_strength,
    candidate_bits,
    coarse_to_fine_mutation_strength,
    mutate_interaction_aware_candidate,
    mutate_joint_aware_candidate,
    selected_candidate_metadata,
)


class FakeWeight:
    def numel(self) -> int:
        return 16


class FakeLayer:
    def __init__(self) -> None:
        self.weight = FakeWeight()


class FakeModel:
    def __init__(self, layer_names: list[str]) -> None:
        self.layers = {name: FakeLayer() for name in layer_names}

    def get_submodule(self, layer_name: str) -> FakeLayer:
        return self.layers[layer_name]


class JointAwareMutationTest(unittest.TestCase):
    def test_coarse_to_fine_strength_uses_equal_schedule_stages(self) -> None:
        strengths = [
            coarse_to_fine_mutation_strength(generation, 20, 3, 1)
            for generation in range(1, 21)
        ]
        self.assertEqual(strengths[:6], [3] * 6)
        self.assertEqual(strengths[6:13], [2] * 7)
        self.assertEqual(strengths[13:], [1] * 7)

    def test_coarse_to_fine_strength_rejects_invalid_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            coarse_to_fine_mutation_strength(1, 0, 3, 1)
        with self.assertRaisesRegex(ValueError, "search range"):
            coarse_to_fine_mutation_strength(0, 20, 3, 1)
        with self.assertRaisesRegex(ValueError, "greater than"):
            coarse_to_fine_mutation_strength(1, 20, 1, 2)

    def test_adaptive_mutation_strength_increases_at_patience_boundaries(self) -> None:
        self.assertEqual(adaptive_mutation_strength(0, 3, 3), 1)
        self.assertEqual(adaptive_mutation_strength(2, 3, 3), 1)
        self.assertEqual(adaptive_mutation_strength(3, 3, 3), 2)
        self.assertEqual(adaptive_mutation_strength(6, 3, 3), 3)
        self.assertEqual(adaptive_mutation_strength(20, 3, 3), 3)

    def test_adaptive_mutation_strength_rejects_invalid_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            adaptive_mutation_strength(-1, 3, 3)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            adaptive_mutation_strength(0, 0, 3)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            adaptive_mutation_strength(0, 3, 0)

    def test_selected_candidate_metadata_follows_survivors(self) -> None:
        candidates = [
            {"drop": {"attn": [False]}, "quant": [[2]]},
            {"drop": {"attn": [True]}, "quant": [[3]]},
            {"drop": {"attn": [False]}, "quant": [[4]]},
        ]
        mutation_types = ["depth", "joint_aware", "quantization"]

        selected = [candidates[2], candidates[0]]

        self.assertEqual(
            selected_candidate_metadata(
                selected,
                candidates,
                mutation_types,
            ),
            ["quantization", "depth"],
        )

    def test_selected_candidate_metadata_rejects_misaligned_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            selected_candidate_metadata([{"candidate": 1}], [], ["depth"])

    def test_coupled_mutation_preserves_budgets_and_touches_restored_layer(self) -> None:
        layer_names = [
            f"model.layers.{layer_id}.self_attn.q_proj"
            for layer_id in range(4)
        ]
        grouped_layer_names = [layer_names]
        model = FakeModel(layer_names)
        candidate = {
            "drop": {
                "attn": [False, False, True, True],
                "mlp": [False, False, True, True],
            },
            "quant": [[3, 3, 3, 3]],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            quant_root = Path(temp_dir)
            for layer_name in layer_names:
                module_dir = quant_root / layer_name
                module_dir.mkdir(parents=True)
                for bitwidth in (2, 3, 4):
                    (module_dir / f"{bitwidth}.pth").touch()

            random.seed(7)
            offspring = mutate_joint_aware_candidate(
                model,
                grouped_layer_names,
                str(quant_root),
                candidate,
                target_bitwidth=3.0,
                step_size=1,
            )

        self.assertEqual(sum(offspring["drop"]["attn"]), 2)
        self.assertEqual(sum(offspring["drop"]["mlp"]), 2)
        self.assertNotEqual(offspring["drop"], candidate["drop"])
        self.assertNotEqual(offspring["quant"], candidate["quant"])

        restored = [
            layer_id
            for layer_id in range(4)
            if candidate["drop"]["attn"][layer_id]
            and not offspring["drop"]["attn"][layer_id]
        ]
        self.assertEqual(len(restored), 1)
        self.assertNotEqual(offspring["quant"][0][restored[0]], 3)

        active_weights = 2 * 16
        active_bits = candidate_bits(
            model,
            grouped_layer_names,
            offspring["quant"],
            offspring["drop"],
        )
        self.assertEqual(active_bits / active_weights, 3.0)

    def test_interaction_aware_mutation_changes_depth_and_quantization(self) -> None:
        layer_names = [
            f"model.layers.{layer_id}.self_attn.q_proj"
            for layer_id in range(4)
        ]
        grouped_layer_names = [layer_names]
        model = FakeModel(layer_names)
        candidate = {
            "drop": {
                "attn": [False, False, True, True],
                "mlp": [False, False, True, True],
            },
            "quant": [[3, 3, 3, 3]],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            quant_root = Path(temp_dir)
            for layer_name in layer_names:
                module_dir = quant_root / layer_name
                module_dir.mkdir(parents=True)
                for bitwidth in (2, 3, 4):
                    (module_dir / f"{bitwidth}.pth").touch()

            random.seed(11)
            offspring, details = mutate_interaction_aware_candidate(
                model,
                grouped_layer_names,
                str(quant_root),
                candidate,
                target_bitwidth=3.0,
                step_size=1,
                drop_entire_block=True,
                max_drop_mutations=1,
                quant_mutations=1,
            )

        self.assertEqual(sum(offspring["drop"]["attn"]), 2)
        self.assertEqual(sum(offspring["drop"]["mlp"]), 2)
        self.assertNotEqual(offspring["drop"], candidate["drop"])
        self.assertNotEqual(offspring["quant"], candidate["quant"])
        self.assertGreater(details["depth_mask_entries_changed"], 0)
        self.assertGreater(details["quant_assignments_changed"], 0)
        self.assertTrue(
            details["preferred_quant_exchange_used"]
            or details["fallback_quant_exchange_used"]
        )

        active_weights = 2 * 16
        active_bits = candidate_bits(
            model,
            grouped_layer_names,
            offspring["quant"],
            offspring["drop"],
        )
        self.assertEqual(active_bits / active_weights, 3.0)


if __name__ == "__main__":
    unittest.main()
