import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evo_joint_search import (
    add_persistent_parents_for_elitism,
    candidate_is_duplicate,
    component_crossover,
    crossover_is_enabled,
    effective_survivors_per_selection,
    select_distinct_parents,
    select_mutation_parent,
    try_component_crossover,
    unique_candidate_count,
    validate_crossover_configuration,
    validate_persistent_population,
)
from src.sequential_search import (
    validate_active_quant_budget,
    validate_depth_counts,
)


def make_candidate(drop_ids, quant):
    drop = [index in drop_ids for index in range(4)]
    return {
        "drop": {"attn": drop, "mlp": copy.deepcopy(drop)},
        "quant": [list(quant)],
    }


class ComponentCrossoverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layer_names = [
            f"model.layers.{layer_id}.self_attn.q_proj"
            for layer_id in range(4)
        ]
        self.grouped_layer_names = [self.layer_names]
        self.parent_a = make_candidate({0, 1}, [3, 3, 3, 3])
        self.parent_b = make_candidate({2, 3}, [2, 4, 2, 2])

    def create_quant_database(self, root: Path, bitwidths=(2, 3, 4)) -> None:
        for layer_name in self.layer_names:
            module_dir = root / layer_name
            module_dir.mkdir(parents=True)
            for bitwidth in bitwidths:
                (module_dir / f"{bitwidth}.pth").touch()

    def test_component_crossover_constructs_either_component_pair(self) -> None:
        child_ab, details_ab = component_crossover(
            self.parent_a,
            self.parent_b,
            use_depth_from_a=True,
        )
        self.assertEqual(child_ab["drop"], self.parent_a["drop"])
        self.assertEqual(child_ab["quant"], self.parent_b["quant"])
        self.assertEqual(details_ab, {"depth_parent": "a", "quant_parent": "b"})

        child_ba, details_ba = component_crossover(
            self.parent_a,
            self.parent_b,
            use_depth_from_a=False,
        )
        self.assertEqual(child_ba["drop"], self.parent_b["drop"])
        self.assertEqual(child_ba["quant"], self.parent_a["quant"])
        self.assertEqual(details_ba, {"depth_parent": "b", "quant_parent": "a"})

    def test_parents_are_unchanged_and_child_is_deep_copied(self) -> None:
        before_a = copy.deepcopy(self.parent_a)
        before_b = copy.deepcopy(self.parent_b)
        child, _ = component_crossover(
            self.parent_a,
            self.parent_b,
            use_depth_from_a=True,
        )

        self.assertIsNot(child["drop"], self.parent_a["drop"])
        self.assertIsNot(child["quant"], self.parent_b["quant"])
        child["drop"]["attn"][0] = False
        child["quant"][0][0] = 4
        self.assertEqual(self.parent_a, before_a)
        self.assertEqual(self.parent_b, before_b)

    def test_depth_counts_remain_valid(self) -> None:
        child, _ = component_crossover(
            self.parent_a,
            self.parent_b,
            use_depth_from_a=True,
        )
        self.assertTrue(validate_depth_counts(child["drop"], 4, 2, True))

    def test_active_budget_repair_is_applied_and_classified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            quant_root = Path(temp_dir)
            self.create_quant_database(quant_root)
            random.seed(7)
            child, details = try_component_crossover(
                self.parent_a,
                self.parent_b,
                grouped_layer_names=self.grouped_layer_names,
                quant_weights_path=str(quant_root),
                target_bitwidth=3.0,
                total_blocks=4,
                blocks_to_remove=2,
                active_quant_budget=True,
                drop_entire_block=True,
                use_depth_from_a=True,
            )

        self.assertIsNotNone(child)
        self.assertGreater(details["repair_changed_gene_count"], 0)
        self.assertEqual(details["classification"], "component crossover + repair")
        self.assertTrue(
            validate_active_quant_budget(
                self.grouped_layer_names,
                child["quant"],
                child["drop"],
                3.0,
            )
        )
        self.assertEqual(child["drop"], self.parent_a["drop"])
        self.assertEqual(self.parent_b["quant"], [[2, 4, 2, 2]])

    def test_infeasible_crossover_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            quant_root = Path(temp_dir)
            self.create_quant_database(quant_root, bitwidths=(2,))
            child, details = try_component_crossover(
                self.parent_a,
                self.parent_b,
                grouped_layer_names=self.grouped_layer_names,
                quant_weights_path=str(quant_root),
                target_bitwidth=3.0,
                total_blocks=4,
                blocks_to_remove=2,
                active_quant_budget=True,
                drop_entire_block=True,
                use_depth_from_a=True,
            )

        self.assertIsNone(child)
        self.assertIn("Unable to restore", details["rejection_reason"])

    def test_duplicate_crossover_candidate_is_detected(self) -> None:
        child, _ = component_crossover(
            self.parent_a,
            self.parent_b,
            use_depth_from_a=True,
        )
        self.assertTrue(candidate_is_duplicate(child, [copy.deepcopy(child)]))

    def test_two_distinct_parents_are_selected(self) -> None:
        population = [
            self.parent_a,
            self.parent_b,
            make_candidate({0, 2}, [2, 3, 3, 4]),
            make_candidate({1, 3}, [4, 3, 3, 2]),
        ]
        random.seed(3)
        parent_a, parent_b = select_distinct_parents(population)
        self.assertNotEqual(parent_a, parent_b)
        self.assertIn(parent_a, population)
        self.assertIn(parent_b, population)

    def test_final_stage_elitism_includes_all_persistent_parents(self) -> None:
        persistent_population = [
            self.parent_a,
            self.parent_b,
            make_candidate({0, 2}, [2, 3, 3, 4]),
            make_candidate({1, 3}, [4, 3, 3, 2]),
        ]
        stage_candidates = [
            make_candidate({0, 3}, [2, 2, 4, 4]),
            make_candidate({1, 2}, [4, 4, 2, 2]),
        ]
        pool, metadata = add_persistent_parents_for_elitism(
            stage_candidates,
            ["depth", "quantization"],
            persistent_population,
        )

        for parent in persistent_population:
            self.assertIn(parent, pool)
        self.assertEqual(metadata.count("parent"), 4)
        self.assertEqual(unique_candidate_count(pool), 6)

    def test_requested_population_size_is_enforced(self) -> None:
        population = [
            self.parent_a,
            self.parent_b,
            make_candidate({0, 2}, [2, 3, 3, 4]),
            make_candidate({1, 3}, [4, 3, 3, 2]),
        ]
        validate_persistent_population(
            population,
            4,
            context="test final selection",
        )
        with self.assertRaisesRegex(RuntimeError, "Unable to retain 4 unique"):
            validate_persistent_population(
                population[:3],
                4,
                context="test final selection",
            )

    def test_population_four_rewrites_only_final_schedule_stage(self) -> None:
        self.assertEqual(
            effective_survivors_per_selection([8, 2, 1], 4),
            [8, 2, 4],
        )

    def test_crossover_is_rejected_with_sequential_modes(self) -> None:
        args = SimpleNamespace(
            population_size=4,
            crossover_probability=0.25,
            initially_generated=16,
            sequential_mode="depth_to_joint_warm",
        )
        with self.assertRaisesRegex(ValueError, "sequential_mode none"):
            validate_crossover_configuration(args)

    def test_legacy_single_parent_configuration_preserves_schedule_and_rng(self) -> None:
        self.assertFalse(crossover_is_enabled(1, 0.0))
        self.assertEqual(
            effective_survivors_per_selection([8, 2, 1], 1),
            [8, 2, 1],
        )
        random.seed(19)
        before = random.getstate()
        selected = select_mutation_parent([self.parent_a])
        after = random.getstate()
        self.assertIs(selected, self.parent_a)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
