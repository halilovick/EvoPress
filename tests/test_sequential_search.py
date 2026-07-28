import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evo_joint_search import (
    make_initial_quant_state,
    make_random_drop_state,
    mutate_interaction_aware_candidate,
    mutate_quant_state,
)
from src.sequential_search import (
    Stage1Import,
    SequentialSearchError,
    build_sequential_summary_metadata,
    enumerate_legal_fixed_quant_depth_swaps,
    generate_exact_feasible_depth_states,
    load_stage1_depth_candidate,
    load_stage1_quant_candidate,
    mutate_fixed_quant_depth_candidate,
    resolve_stage1_artifacts,
    validate_active_quant_budget,
    validate_depth_counts,
    validate_frozen_component,
    validate_sequential_cli,
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


def quant_names(num_layers: int = 4) -> list[str]:
    return [
        f"model.layers.{layer_id}.self_attn.q_proj" for layer_id in range(num_layers)
    ]


def write_quant_database(root: Path, names: list[str]) -> None:
    for name in names:
        module_dir = root / name
        module_dir.mkdir(parents=True)
        for bitwidth in (2, 3, 4):
            (module_dir / f"{bitwidth}.pth").touch()


def write_stage1_run(
    run_dir: Path,
    candidate: dict,
    *,
    search_type: str,
    group_rule: str | None = None,
    target_bitwidth: float | None = None,
    drop_entire_block: bool = False,
) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "final_candidate.json").write_text(
        json.dumps(candidate),
        encoding="utf-8",
    )
    compression = {
        "drop_entire_block": drop_entire_block,
        "target_average_bitwidth": target_bitwidth,
    }
    if group_rule is not None:
        compression["group_rule"] = group_rule
    summary = {
        "model_name": "fixture/model",
        "search_type": search_type,
        "compression_config": compression,
        "depth_statistics": {"num_layers": 4},
        "artifacts": {"candidate_path": "remote/final_candidate.json"},
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )


def depth_candidate() -> dict:
    drop = {
        "attn": [False, True, False, True],
        "mlp": [True, False, True, False],
    }
    return {
        "candidate_type": "depth_only",
        "attention_mask": [int(value) for value in drop["attn"]],
        "mlp_mask": [int(value) for value in drop["mlp"]],
        "bitwidth_by_module": {},
        "candidate_vector_raw": drop,
    }


def quant_candidate(names: list[str], levels: list[int]) -> dict:
    return {
        "candidate_type": "quant_only",
        "attention_mask": [0, 0, 0, 0],
        "mlp_mask": [0, 0, 0, 0],
        "bitwidth_by_module": dict(zip(names, levels)),
        "candidate_vector_raw": [levels],
    }


def sequential_args(mode: str, **overrides):
    values = {
        "sequential_mode": mode,
        "stage1_run_dir": "/tmp/stage1",
        "stage1_candidate": None,
        "sequential_quant_initialization_policy": "strict",
        "max_initialization_attempts": 1000,
        "max_offspring_attempts": 100,
        "joint_mutation_mode": "standard",
        "joint_aware_mutation": False,
        "coarse_to_fine_mutation": False,
        "adaptive_mutation": False,
        "active_quant_budget": mode.startswith("quant_to_"),
        "group_rule": "size" if mode.startswith("quant_to_") else "none",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class CandidateLoadingTest(unittest.TestCase):
    def test_depth_candidate_loads_two_masks_and_is_deep_copied(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "depth"
            source = depth_candidate()
            write_stage1_run(run_dir, source, search_type="depth_only")
            artifacts = resolve_stage1_artifacts(run_dir, None)
            loaded = load_stage1_depth_candidate(
                artifacts,
                expected_model_name="fixture/model",
                num_layers=4,
                drop_count=2,
                drop_entire_block=False,
            )

            self.assertEqual(loaded.component, source["candidate_vector_raw"])
            loaded.component["attn"][0] = True
            self.assertFalse(artifacts.candidate["candidate_vector_raw"]["attn"][0])

    def test_incompatible_depth_layer_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "depth"
            write_stage1_run(
                run_dir,
                depth_candidate(),
                search_type="depth_only",
            )
            artifacts = resolve_stage1_artifacts(run_dir, None)
            with self.assertRaisesRegex(SequentialSearchError, "layer-count"):
                load_stage1_depth_candidate(
                    artifacts,
                    expected_model_name="fixture/model",
                    num_layers=5,
                    drop_count=2,
                    drop_entire_block=False,
                )

    def test_quant_candidate_maps_by_module_name(self) -> None:
        names = quant_names()
        levels = [2, 2, 4, 4]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            quant_root = root / "quant_db"
            write_quant_database(quant_root, names)
            run_dir = root / "quant"
            write_stage1_run(
                run_dir,
                quant_candidate(names, levels),
                search_type="quant_only",
                group_rule="size",
                target_bitwidth=3.0,
            )
            artifacts = resolve_stage1_artifacts(None, run_dir / "final_candidate.json")
            loaded = load_stage1_quant_candidate(
                artifacts,
                expected_model_name="fixture/model",
                num_layers=4,
                grouped_layer_names=[[names[2], names[0], names[3], names[1]]],
                group_rule="size",
                target_bitwidth=3.0,
                quant_weights_path=quant_root,
            )

            self.assertEqual(loaded.component, [[4, 2, 4, 2]])
            loaded.component[0][0] = 3
            self.assertEqual(
                artifacts.candidate["bitwidth_by_module"][names[2]],
                4,
            )

    def test_missing_quant_module_is_rejected(self) -> None:
        names = quant_names()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            quant_root = root / "quant_db"
            write_quant_database(quant_root, names)
            candidate = quant_candidate(names[:-1], [2, 2, 4])
            candidate["candidate_vector_raw"] = [[2, 2, 4]]
            run_dir = root / "quant"
            write_stage1_run(
                run_dir,
                candidate,
                search_type="quant_only",
                group_rule="size",
                target_bitwidth=3.0,
            )
            artifacts = resolve_stage1_artifacts(run_dir, None)
            with self.assertRaisesRegex(SequentialSearchError, "missing_modules"):
                load_stage1_quant_candidate(
                    artifacts,
                    expected_model_name="fixture/model",
                    num_layers=4,
                    grouped_layer_names=[names],
                    group_rule="size",
                    target_bitwidth=3.0,
                    quant_weights_path=quant_root,
                )

    def test_grouping_mismatch_is_rejected(self) -> None:
        names = quant_names()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            quant_root = root / "quant_db"
            write_quant_database(quant_root, names)
            run_dir = root / "quant"
            write_stage1_run(
                run_dir,
                quant_candidate(names, [2, 2, 4, 4]),
                search_type="quant_only",
                group_rule="none",
                target_bitwidth=3.0,
            )
            artifacts = resolve_stage1_artifacts(run_dir, None)
            with self.assertRaisesRegex(SequentialSearchError, "grouping mismatch"):
                load_stage1_quant_candidate(
                    artifacts,
                    expected_model_name="fixture/model",
                    num_layers=4,
                    grouped_layer_names=[names],
                    group_rule="size",
                    target_bitwidth=3.0,
                    quant_weights_path=quant_root,
                )


class ExactFrozenQuantDepthTest(unittest.TestCase):
    def setUp(self) -> None:
        self.names = quant_names()
        self.grouped_names = [self.names]
        self.quant = [[2, 2, 4, 4]]

    def test_exact_initial_masks_preserve_counts_profile_and_budget(self) -> None:
        states = generate_exact_feasible_depth_states(
            self.grouped_names,
            self.quant,
            num_layers=4,
            drop_count=2,
            target_bitwidth=3.0,
            drop_entire_block=False,
            requested_candidates=4,
            max_states=10000,
        )
        self.assertEqual(len(states), 4)
        for state in states:
            validate_depth_counts(state, 4, 2, False)
            validate_active_quant_budget(
                self.grouped_names,
                self.quant,
                state,
                3.0,
            )

    def test_legal_swaps_preserve_contribution_and_frozen_quant(self) -> None:
        candidate = {
            "drop": {
                "attn": [True, False, True, False],
                "mlp": [True, True, False, False],
            },
            "quant": copy.deepcopy(self.quant),
        }
        legal = enumerate_legal_fixed_quant_depth_swaps(
            candidate["drop"],
            self.grouped_names,
            candidate["quant"],
            target_bitwidth=3.0,
            drop_entire_block=False,
        )
        self.assertIn(("attn", 1, 0), legal)
        self.assertNotIn(("attn", 3, 0), legal)

        random.seed(5)
        offspring, details = mutate_fixed_quant_depth_candidate(
            candidate,
            self.grouped_names,
            target_bitwidth=3.0,
            drop_entire_block=False,
            max_mutations=1,
        )
        self.assertIsNotNone(offspring)
        self.assertGreater(details["legal_swap_count"], 0)
        self.assertEqual(offspring["quant"], candidate["quant"])
        self.assertNotEqual(offspring["drop"], candidate["drop"])
        validate_active_quant_budget(
            self.grouped_names,
            offspring["quant"],
            offspring["drop"],
            3.0,
        )

    def test_no_feasible_initial_mask_has_clear_error(self) -> None:
        with self.assertRaisesRegex(
            SequentialSearchError,
            "No exact depth mask is feasible",
        ):
            generate_exact_feasible_depth_states(
                [quant_names(2)],
                [[2, 4]],
                num_layers=2,
                drop_count=1,
                target_bitwidth=3.0,
                drop_entire_block=True,
                requested_candidates=1,
                max_states=1000,
            )

    def test_no_legal_mutation_returns_controlled_failure(self) -> None:
        names = quant_names(3)
        candidate = {
            "drop": {
                "attn": [False, True, False],
                "mlp": [False, True, False],
            },
            "quant": [[2, 3, 4]],
        }
        offspring, details = mutate_fixed_quant_depth_candidate(
            candidate,
            [names],
            target_bitwidth=3.0,
            drop_entire_block=True,
        )
        self.assertIsNone(offspring)
        self.assertEqual(details["legal_swap_count"], 0)
        self.assertEqual(details["reason"], "no_legal_fixed_quant_depth_swap")


class SequentialModeSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.names = quant_names()
        self.grouped_names = [self.names]
        self.model = FakeModel(self.names)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.quant_root = Path(self.temp_dir.name)
        write_quant_database(self.quant_root, self.names)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_depth_to_quant_frozen(self) -> None:
        imported_depth = {
            "attn": [True, True, False, False],
            "mlp": [True, True, False, False],
        }
        parent = {"drop": copy.deepcopy(imported_depth), "quant": [[3, 3, 3, 3]]}
        random.seed(4)
        child = copy.deepcopy(parent)
        child["quant"] = mutate_quant_state(
            self.model,
            self.grouped_names,
            str(self.quant_root),
            child["quant"],
            drop_state=child["drop"],
        )
        validate_frozen_component(child, imported_depth, "depth")
        self.assertNotEqual(child["quant"], parent["quant"])
        validate_active_quant_budget(
            self.grouped_names,
            child["quant"],
            child["drop"],
            3.0,
        )

    def test_default_joint_initialization_shape_is_unchanged(self) -> None:
        random.seed(2)
        candidate = {
            "drop": make_random_drop_state(4, 2, False),
            "quant": make_initial_quant_state(
                self.model,
                self.grouped_names,
                str(self.quant_root),
                3.0,
            ),
        }
        self.assertEqual(set(candidate), {"drop", "quant"})
        validate_depth_counts(candidate["drop"], 4, 2, False)
        self.assertEqual(candidate["quant"], [[3, 3, 3, 3]])

    def test_depth_to_joint_warm_allows_both_components(self) -> None:
        imported_depth = {
            "attn": [True, True, False, False],
            "mlp": [True, True, False, False],
        }
        parent = {"drop": copy.deepcopy(imported_depth), "quant": [[3, 3, 3, 3]]}
        validate_frozen_component(parent, imported_depth, "depth")
        random.seed(11)
        child, _ = mutate_interaction_aware_candidate(
            self.model,
            self.grouped_names,
            str(self.quant_root),
            parent,
            target_bitwidth=3.0,
            step_size=1,
            drop_entire_block=True,
            max_drop_mutations=1,
            quant_mutations=1,
        )
        self.assertNotEqual(child["drop"], parent["drop"])
        self.assertNotEqual(child["quant"], parent["quant"])
        validate_active_quant_budget(
            self.grouped_names,
            child["quant"],
            child["drop"],
            3.0,
        )

    def test_quant_to_depth_frozen(self) -> None:
        imported_quant = [[2, 2, 4, 4]]
        drop = generate_exact_feasible_depth_states(
            self.grouped_names,
            imported_quant,
            num_layers=4,
            drop_count=2,
            target_bitwidth=3.0,
            drop_entire_block=True,
            requested_candidates=1,
            max_states=10000,
        )[0]
        parent = {"drop": drop, "quant": copy.deepcopy(imported_quant)}
        random.seed(3)
        child, _ = mutate_fixed_quant_depth_candidate(
            parent,
            self.grouped_names,
            target_bitwidth=3.0,
            drop_entire_block=True,
        )
        self.assertIsNotNone(child)
        validate_frozen_component(child, imported_quant, "quantization")
        validate_active_quant_budget(
            self.grouped_names,
            child["quant"],
            child["drop"],
            3.0,
        )

    def test_quant_to_joint_warm_preserves_import_then_allows_quant_change(
        self,
    ) -> None:
        imported_quant = [[2, 2, 4, 4]]
        drop = generate_exact_feasible_depth_states(
            self.grouped_names,
            imported_quant,
            num_layers=4,
            drop_count=2,
            target_bitwidth=3.0,
            drop_entire_block=True,
            requested_candidates=1,
            max_states=10000,
        )[0]
        parent = {"drop": drop, "quant": copy.deepcopy(imported_quant)}
        validate_frozen_component(parent, imported_quant, "quantization")
        random.seed(13)
        child, _ = mutate_interaction_aware_candidate(
            self.model,
            self.grouped_names,
            str(self.quant_root),
            parent,
            target_bitwidth=3.0,
            step_size=1,
            drop_entire_block=True,
            max_drop_mutations=1,
            quant_mutations=1,
        )
        self.assertNotEqual(child["quant"], imported_quant)
        validate_active_quant_budget(
            self.grouped_names,
            child["quant"],
            child["drop"],
            3.0,
        )


class SequentialCliValidationTest(unittest.TestCase):
    def test_frozen_modes_reject_interaction_aware(self) -> None:
        with self.assertRaisesRegex(SequentialSearchError, "interaction-aware"):
            validate_sequential_cli(
                sequential_args(
                    "depth_to_quant_frozen",
                    joint_mutation_mode="interaction_aware",
                )
            )

    def test_quant_frozen_rejects_repair_policy(self) -> None:
        with self.assertRaisesRegex(SequentialSearchError, "allowed only"):
            validate_sequential_cli(
                sequential_args(
                    "quant_to_depth_frozen",
                    sequential_quant_initialization_policy="repair",
                )
            )

    def test_missing_stage_one_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(SequentialSearchError, "exactly one"):
            validate_sequential_cli(
                sequential_args(
                    "depth_to_joint_warm",
                    stage1_run_dir=None,
                )
            )

    def test_mode_none_rejects_stage_one_options(self) -> None:
        with self.assertRaisesRegex(SequentialSearchError, "non-'none'"):
            validate_sequential_cli(sequential_args("none"))

    def test_quant_first_requires_active_size_grouping(self) -> None:
        with self.assertRaisesRegex(SequentialSearchError, "active_quant_budget"):
            validate_sequential_cli(
                sequential_args(
                    "quant_to_joint_warm",
                    active_quant_budget=False,
                )
            )
        with self.assertRaisesRegex(SequentialSearchError, "group_rule size"):
            validate_sequential_cli(
                sequential_args(
                    "quant_to_joint_warm",
                    group_rule="none",
                )
            )

    def test_mode_none_without_options_preserves_default_path(self) -> None:
        validate_sequential_cli(
            sequential_args(
                "none",
                stage1_run_dir=None,
                stage1_candidate=None,
            )
        )

    def test_toy_summary_metadata_for_all_four_modes(self) -> None:
        initial = {
            "drop": {"attn": [True, False], "mlp": [True, False]},
            "quant": [[2, 4]],
        }
        final_by_mode = {
            "depth_to_quant_frozen": {
                "drop": copy.deepcopy(initial["drop"]),
                "quant": [[3, 3]],
            },
            "depth_to_joint_warm": {
                "drop": {"attn": [False, True], "mlp": [False, True]},
                "quant": [[3, 3]],
            },
            "quant_to_depth_frozen": {
                "drop": {"attn": [False, True], "mlp": [False, True]},
                "quant": copy.deepcopy(initial["quant"]),
            },
            "quant_to_joint_warm": {
                "drop": {"attn": [False, True], "mlp": [False, True]},
                "quant": [[3, 3]],
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            for mode, final in final_by_mode.items():
                imported_component = (
                    copy.deepcopy(initial["drop"])
                    if mode.startswith("depth_to_")
                    else copy.deepcopy(initial["quant"])
                )
                stage1 = Stage1Import(
                    component=imported_component,
                    component_hash="fixture-hash",
                    candidate_path="/tmp/final_candidate.json",
                    summary_path="/tmp/run_summary.json",
                    run_dir="/tmp",
                    search_type=(
                        "depth_only" if mode.startswith("depth_to_") else "quant_only"
                    ),
                    model_name="fixture/model",
                    source_group_rule="size",
                    source_target_bitwidth=3.0,
                )
                metadata = build_sequential_summary_metadata(
                    mode=mode,
                    stage1_import=stage1,
                    quant_initialization_policy="strict",
                    initial_repair_changed_gene_names=[],
                    initial_candidate_count=2,
                    initial_parent=initial,
                    final_parent=final,
                    initial_fixed_quant_legal_swap_count=2,
                    final_fixed_quant_legal_swap_count=1,
                    active_budget_valid=True,
                    depth_counts_valid=True,
                )
                summary_path = Path(temp_dir) / f"{mode}_run_summary.json"
                summary_path.write_text(json.dumps(metadata), encoding="utf-8")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self.assertEqual(summary["sequential_mode"], mode)
                self.assertEqual(summary["stage1_candidate_hash"], "fixture-hash")
                self.assertTrue(summary["active_budget_valid"])
                if mode == "depth_to_quant_frozen":
                    self.assertTrue(summary["frozen_depth_unchanged"])
                if mode == "quant_to_depth_frozen":
                    self.assertTrue(summary["frozen_quant_unchanged"])


if __name__ == "__main__":
    unittest.main()
