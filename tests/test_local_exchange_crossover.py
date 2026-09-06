"""CPU-only local-exchange tests; no model downloads or search evaluation."""

import ast
import copy
import csv
import inspect
import json
import random
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import evo_joint_search as search
from src.compression_budget import candidate_compression_cost, validate_exact_budget
from src.search_checkpoint import (
    load_search_checkpoint,
    save_search_checkpoint,
    restore_rng_state,
)


class LocalExchangeCrossoverTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        # Every operator/offspring-loop test fails immediately if repair is used.
        for name in (
            "repair_depth_counts",
            "repair_active_quant_budget",
            "repair_quant_state_to_budget",
        ):
            self.stack.enter_context(
                patch.object(
                    search, name, side_effect=AssertionError(f"repair called: {name}")
                )
            )
        self.model = torch.nn.Module()
        self.model.model = torch.nn.Module()
        self.model.model.layers = torch.nn.ModuleList()
        for _ in range(4):
            block = torch.nn.Module()
            block.self_attn = torch.nn.Module()
            block.self_attn.q_proj = torch.nn.Linear(4, 4, bias=False)
            block.self_attn.k_proj = torch.nn.Linear(4, 2, bias=False)
            block.mlp = torch.nn.Module()
            block.mlp.up_proj = torch.nn.Linear(4, 8, bias=False)
            self.model.model.layers.append(block)
        self.names = [[f"model.layers.{i}.self_attn.q_proj" for i in range(4)]]
        self.cost_kwargs = {
            "attention_module_names": [f"model.layers.{i}.self_attn" for i in range(4)],
            "mlp_module_names": [f"model.layers.{i}.mlp" for i in range(4)],
        }
        self.database()

    def database(self, levels=(2, 3, 4)):
        for group in self.names:
            for name in group:
                directory = self.root / name
                directory.mkdir(exist_ok=True)
                for level in levels:
                    (directory / f"{level}.pth").touch()

    def candidate(self, attn=(), mlp=(), quant=(3, 3, 3, 3)):
        return {
            "drop": {
                "attn": [i in attn for i in range(4)],
                "mlp": [i in mlp for i in range(4)],
            },
            "quant": [list(quant)],
        }

    def propose(self, base, donor, **kwargs):
        options = dict(
            model=self.model,
            grouped_layer_names=self.names,
            quant_weights_path=str(self.root),
            target_bitwidth=3.0,
            total_blocks=4,
            blocks_to_remove=sum(base["drop"]["attn"]),
            active_quant_budget=True,
            budget_cost_kwargs=self.cost_kwargs,
        )
        options.update(kwargs)
        return search.try_local_exchange_crossover(base, donor, **options)

    def cost(self, candidate):
        return candidate_compression_cost(
            self.model, candidate, grouped_layer_names=self.names, **self.cost_kwargs
        )

    def test_attention_exchange_is_local_and_parents_are_deep_copied(self):
        base, donor = self.candidate((0,), (3,)), self.candidate((1,), (3,))
        before = copy.deepcopy((base, donor))
        child, details = self.propose(base, donor)
        self.assertEqual(child["drop"]["attn"], donor["drop"]["attn"])
        self.assertEqual(child["drop"]["mlp"], base["drop"]["mlp"])
        self.assertEqual(child["quant"], base["quant"])
        self.assertTrue(search.validate_depth_counts(child["drop"], 4, 1, False))
        self.assertEqual(details["local_exchange_type"], "attention")
        self.assertEqual(details["changed_gene_count"], 2)
        self.assertEqual(details["donor_distance_reduction"], 2)
        self.assertEqual(details["repair_changed_gene_count"], 0)
        for parent in (base, donor):
            self.assertIsNot(child["drop"], parent["drop"])
            self.assertIsNot(child["drop"]["mlp"], parent["drop"]["mlp"])
            self.assertIsNot(child["quant"][0], parent["quant"][0])
        child["drop"]["mlp"][0] = True
        child["quant"][0][0] = 2
        self.assertEqual((base, donor), before)

    def test_mlp_exchange_changes_only_mlp(self):
        base, donor = self.candidate((3,), (0,)), self.candidate((3,), (1,))
        child, details = self.propose(base, donor)
        self.assertEqual(child["drop"]["mlp"], donor["drop"]["mlp"])
        self.assertEqual(child["drop"]["attn"], base["drop"]["attn"])
        self.assertEqual(child["quant"], base["quant"])
        self.assertTrue(search.validate_depth_counts(child["drop"], 4, 1, False))
        self.assertEqual(details["local_exchange_type"], "mlp")
        self.assertEqual(details["donor_distance_reduction"], 2)

    def test_quantization_exchange_is_adjacent_active_and_cost_neutral(self):
        base = self.candidate((3,), (3,), (2, 4, 3, 3))
        donor = self.candidate((3,), (3,), (4, 2, 3, 3))
        before = copy.deepcopy((base, donor))
        child, details = self.propose(base, donor, step_size=7)
        self.assertEqual(child["quant"], [[3, 3, 3, 3]])
        self.assertEqual(child["drop"], base["drop"])
        self.assertEqual((base, donor), before)
        self.assertEqual(self.cost(child), self.cost(base))
        self.assertTrue(
            search.validate_active_quant_budget(
                self.names, child["quant"], child["drop"], 3.0
            )
        )
        self.assertEqual(details["child_distance_from_base"], 2)
        self.assertEqual(
            details["donor_distance_reduction"], 0
        )  # Hamming, not level distance.
        self.assertEqual(details["local_exchange_type"], "quantization")

    def test_noncontiguous_database_levels(self):
        for name in self.names[0]:
            (self.root / name / "3.pth").unlink()
        self.database((6,))
        base, donor = self.candidate(quant=(2, 6, 4, 4)), self.candidate(
            quant=(6, 2, 4, 4)
        )
        child, _ = self.propose(base, donor, target_bitwidth=4.0)
        self.assertEqual(child["quant"], [[4, 4, 4, 4]])
        self.assertEqual(self.cost(child), self.cost(base))

    def test_cached_metadata_avoids_repeated_model_and_database_inspection(self):
        base, donor = self.candidate(quant=(2, 4, 3, 3)), self.candidate(
            quant=(4, 2, 3, 3)
        )
        metadata = search.build_local_exchange_metadata(
            self.model, self.names, str(self.root)
        )
        with patch.object(
            self.model, "get_submodule", side_effect=AssertionError("model lookup")
        ), patch.object(
            search, "available_module_bitwidths", side_effect=AssertionError("DB scan")
        ):
            child, _ = self.propose(base, donor, metadata=metadata)
        self.assertEqual(child["quant"], [[3, 3, 3, 3]])

    def test_no_legal_exchange_ignores_inactive_guidance(self):
        base = self.candidate((0, 1), (0, 1), (2, 4, 3, 3))
        donor = self.candidate((0, 1), (0, 1), (4, 2, 3, 3))
        child, details = self.propose(base, donor)
        self.assertIsNone(child)
        self.assertTrue(details["no_legal_exchange"])
        self.assertEqual(details["rejection_reason"], "no_legal_exchange")
        self.assertEqual(details["repair_changed_gene_count"], 0)

    def test_dropped_donor_quant_gene_is_not_guidance(self):
        base = self.candidate((3,), (3,), (2, 4, 3, 3))
        donor = self.candidate((0,), (3,), (4, 2, 3, 4))
        _, details = self.propose(base, donor)
        # Upward guidance at gene 0 is inactive in donor; no opposite pair remains.
        self.assertEqual(details["legal_move_counts"]["quantization"], 0)

    def test_structural_swap_that_breaks_active_budget_is_not_legal(self):
        base = self.candidate((0,), (3,), (2, 3, 3, 3))
        donor = self.candidate((1,), (3,))
        child, details = self.propose(base, donor)
        self.assertIsNone(child)
        self.assertTrue(details["no_legal_exchange"])
        self.assertEqual(details["legal_move_counts"]["attention"], 0)

    def test_mlp_projection_budget_is_validated_generically(self):
        self.names = [[f"model.layers.{i}.mlp.up_proj" for i in range(4)]]
        self.database()
        base = self.candidate((3,), (0,), (2, 3, 3, 3))
        donor = self.candidate((3,), (1,))
        child, details = self.propose(base, donor)
        self.assertIsNone(child)
        self.assertEqual(details["legal_move_counts"]["mlp"], 0)

    def test_unequal_cost_pair_is_skipped(self):
        self.model.model.layers[0].self_attn.q_proj = torch.nn.Linear(4, 8, bias=False)
        base, donor = self.candidate(quant=(2, 4, 3, 3)), self.candidate(
            quant=(4, 2, 3, 3)
        )
        child, details = self.propose(base, donor, active_quant_budget=False)
        self.assertIsNone(child)
        self.assertTrue(details["no_legal_exchange"])

    def test_unequal_sizes_can_cancel_with_different_available_steps(self):
        self.model.model.layers[0].self_attn.q_proj = torch.nn.Linear(4, 8, bias=False)
        (self.root / self.names[0][1] / "3.pth").unlink()
        base, donor = self.candidate(quant=(2, 4, 3, 3)), self.candidate(
            quant=(4, 2, 3, 3)
        )
        child, _ = self.propose(base, donor, active_quant_budget=False)
        self.assertEqual(child["quant"], [[3, 2, 3, 3]])
        self.assertEqual(
            self.cost(child)["total_cost_bits"], self.cost(base)["total_cost_bits"]
        )

    def test_active_budget_cannot_transfer_cost_between_groups(self):
        self.names = [[name] for name in self.names[0]]
        base = self.candidate()
        donor = self.candidate()
        base["quant"] = [[3], [3], [3], [3]]
        donor["quant"] = [[4], [2], [3], [3]]
        child, details = self.propose(base, donor)
        self.assertIsNone(child)
        self.assertTrue(details["no_legal_exchange"])

    def test_active_budget_has_no_floating_tolerance(self):
        base, donor = self.candidate(quant=(2, 4, 3, 3)), self.candidate(
            quant=(4, 2, 3, 3)
        )
        child, details = self.propose(base, donor, target_bitwidth=3.0000000001)
        self.assertIsNone(child)
        self.assertFalse(details["no_legal_exchange"])
        self.assertIn("not exact", details["rejection_reason"])

    def test_block_exchange_keeps_masks_coupled(self):
        base, donor = self.candidate((0,), (0,)), self.candidate((1,), (1,))
        child, details = self.propose(base, donor, drop_entire_block=True)
        self.assertEqual(child["drop"]["attn"], child["drop"]["mlp"])
        self.assertEqual(child["quant"], base["quant"])
        self.assertEqual(details["local_exchange_type"], "block")
        self.assertEqual(details["changed_gene_count"], 4)
        self.assertTrue(search.validate_depth_counts(child["drop"], 4, 1, True))

    def test_invalid_coupled_masks_are_rejected_without_repair(self):
        child, details = self.propose(
            self.candidate((0,), (1,)),
            self.candidate((2,), (2,)),
            drop_entire_block=True,
        )
        self.assertIsNone(child)
        self.assertFalse(details["no_legal_exchange"])

    def test_types_then_moves_are_chosen_uniformly_with_search_rng(self):
        base = self.candidate((0,), (0,), (3, 3, 2, 4))
        donor = self.candidate((1,), (1,), (3, 3, 4, 2))
        with patch.object(search.random, "choice", wraps=random.choice) as choose:
            child, details = self.propose(base, donor)
        self.assertEqual(choose.call_count, 2)
        self.assertEqual(
            choose.call_args_list[0].args[0], ["attention", "mlp", "quantization"]
        )
        self.assertEqual(
            len(choose.call_args_list[1].args[0]),
            details["legal_move_counts"][details["local_exchange_type"]],
        )
        self.assertEqual(search.joint_genotype_distance(child, base), 2)
        random.seed(51)
        first = self.propose(base, donor)
        random.seed(51)
        self.assertEqual(first, self.propose(base, donor))

    def test_existing_duplicate_logic_still_rejects_donor_and_repeated_child(self):
        base, donor = self.candidate((0,), (3,)), self.candidate((1,), (3,))
        child, _ = self.propose(base, donor)
        self.assertTrue(search.candidate_is_duplicate(child, [base, donor]))
        self.assertTrue(
            search.candidate_is_duplicate(child, [], [copy.deepcopy(child)])
        )

    def test_exact_total_budget_rejects_unequal_structural_cost(self):
        self.model.model.layers[0].self_attn.k_proj = torch.nn.Linear(4, 4, bias=False)
        base, donor = self.candidate((0,), (3,)), self.candidate((1,), (3,))
        child, details = self.propose(
            base,
            donor,
            active_quant_budget=False,
            target_cost_bits=self.cost(base)["total_cost_bits"],
        )
        self.assertIsNone(child)
        self.assertTrue(details["no_legal_exchange"])
        # Equal q_proj bits alone would allow this; dense k_proj cost forbids it.

    def test_exact_total_budget_accepts_neutral_move_with_metadata(self):
        self.cost_kwargs.update(
            group_size=2,
            include_quantization_metadata=True,
            scale_bits=8,
            zero_point_bits=8,
        )
        base, donor = self.candidate(quant=(2, 4, 3, 3)), self.candidate(
            quant=(4, 2, 3, 3)
        )
        target = self.cost(base)["total_cost_bits"]
        child, _ = self.propose(
            base, donor, active_quant_budget=False, target_cost_bits=target
        )
        self.assertTrue(validate_exact_budget(self.cost(child), target))

    def run_offspring_loop(self, base, donor, proposals, exact=False):
        """Execute the production offspring loop, with no initialization/evaluation.

        Extracting this block lets the test cover dispatch, retries, duplicate
        accounting and the generic budget path without loading an LLM or data.
        """
        main_ast = ast.parse(inspect.getsource(search.main)).body[0]
        generation_loop = next(
            node
            for node in main_ast.body
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "generation"
        )
        start = next(
            i
            for i, node in enumerate(generation_loop.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "offspring_list"
                for t in node.targets
            )
        )
        stop = next(
            i
            for i in range(start, len(generation_loop.body))
            if isinstance(generation_loop.body[i], ast.While)
        )
        block = ast.Module(body=generation_loop.body[start : stop + 1], type_ignores=[])
        namespace = dict(vars(search))
        namespace.update(
            args=SimpleNamespace(
                offspring=1,
                max_offspring_attempts=len(proposals),
                crossover_probability=1.0,
                crossover_type="local_exchange",
                crossover_parent_selection="uniform",
                quant_weights_path=str(self.root),
                target_bitwidth=3.0,
                active_quant_budget=not exact,
                step_size=1,
                drop_entire_block=False,
                sequential_mode="none",
            ),
            model=self.model,
            grouped_layer_names=self.names,
            total_blocks=4,
            blocks_to_remove=sum(base["drop"]["attn"]),
            population=[base, donor],
            crossover_enabled=True,
            exact_total_budget=exact,
            budget_cost_kwargs=self.cost_kwargs,
            target_cost_bits=self.cost(base)["total_cost_bits"] if exact else None,
            local_exchange_metadata=search.build_local_exchange_metadata(
                self.model, self.names, str(self.root)
            ),
            exchange_diagnostics_total=search.new_exchange_diagnostics(),
        )
        for counter in (
            "crossover_offspring_attempted",
            "crossover_offspring_accepted",
            "crossover_duplicates",
            "crossover_infeasible_candidates",
            "crossover_repaired_proposals",
            "crossover_repair_changed_gene_count",
            "mutation_offspring_accepted",
        ):
            namespace[f"{counter}_total"] = 0
        with patch.object(
            search, "select_distinct_parents", return_value=(base, donor)
        ), patch.object(search, "try_local_exchange_crossover", side_effect=proposals):
            namespace["select_distinct_parents"] = search.select_distinct_parents
            namespace["try_local_exchange_crossover"] = (
                search.try_local_exchange_crossover
            )
            exec(compile(block, search.__file__, "exec"), namespace)
        return namespace

    def test_main_loop_separates_no_legal_exchange_and_duplicates(self):
        base, donor = self.candidate(quant=(2, 4, 3, 3)), self.candidate(
            quant=(4, 2, 3, 3)
        )
        proposal = self.propose(base, donor)
        no_legal = self.propose(
            self.candidate((0, 1), (0, 1), (2, 4, 3, 3)),
            self.candidate((0, 1), (0, 1), (4, 2, 3, 3)),
        )
        state = self.run_offspring_loop(
            base, donor, [no_legal, (copy.deepcopy(donor), proposal[1]), proposal]
        )
        self.assertEqual(state["crossover_offspring_attempted_total"], 3)
        self.assertEqual(state["crossover_offspring_accepted_total"], 1)
        self.assertEqual(state["crossover_duplicates_total"], 1)
        self.assertEqual(state["crossover_infeasible_candidates_total"], 0)
        self.assertEqual(state["infeasible_candidates"], 0)
        self.assertEqual(state["crossover_repaired_proposals_total"], 0)
        self.assertEqual(state["crossover_repair_changed_gene_count_total"], 0)
        summary = search.summarize_exchange_diagnostics(
            state["exchange_diagnostics_total"]
        )
        self.assertEqual(summary["crossover_no_legal_exchange"], 1)
        self.assertEqual(summary["local_exchange_quantization"], 1)
        self.assertEqual(summary["child_distance_from_base_mean"], 2)
        self.assertEqual(summary["parent_distance_count"], 3)
        self.assertEqual(summary["child_distance_from_base_count"], 1)

    def test_main_exact_budget_path_validates_without_repair(self):
        base, donor = self.candidate(quant=(2, 4, 3, 3)), self.candidate(
            quant=(4, 2, 3, 3)
        )
        child, details = self.propose(
            base,
            donor,
            active_quant_budget=False,
            target_cost_bits=self.cost(base)["total_cost_bits"],
        )
        invalid = copy.deepcopy(child)
        invalid["quant"][0][0] = 4
        state = self.run_offspring_loop(
            base, donor, [(invalid, details), (child, details)], exact=True
        )
        self.assertEqual(state["crossover_infeasible_candidates_total"], 1)
        self.assertEqual(state["crossover_offspring_accepted_total"], 1)
        self.assertEqual(state["crossover_repaired_proposals_total"], 0)
        self.assertEqual(state["crossover_repair_changed_gene_count_total"], 0)
        self.assertEqual(state["offspring_list"], [child])

    def test_generic_budget_repair_is_preserved_for_old_operators_and_mutation(self):
        main_ast = ast.parse(inspect.getsource(search.main)).body[0]
        generation_loop = next(
            node
            for node in main_ast.body
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "generation"
        )
        offspring_loop = next(
            node for node in generation_loop.body if isinstance(node, ast.While)
        )
        budget_check = next(
            node
            for node in offspring_loop.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "exact_total_budget"
        )
        # Keep the production rejection `continue` legal without running search.
        block = ast.fix_missing_locations(
            ast.Module(
                body=[
                    ast.For(
                        target=ast.Name(id="once", ctx=ast.Store()),
                        iter=ast.Tuple(elts=[ast.Constant(None)], ctx=ast.Load()),
                        body=[budget_check],
                        orelse=[],
                    )
                ],
                type_ignores=[],
            )
        )
        for use_crossover, crossover_type in (
            (True, "component"),
            (True, "layer_bundle"),
            (False, "local_exchange"),
        ):
            with self.subTest(
                use_crossover=use_crossover, crossover_type=crossover_type
            ):
                base = self.candidate()
                namespace = dict(vars(search))
                with patch.object(
                    search, "repair_quant_state_to_budget", return_value=base["quant"]
                ) as repair:
                    namespace.update(
                        repair_quant_state_to_budget=repair,
                        exact_total_budget=True,
                        use_crossover=use_crossover,
                        args=SimpleNamespace(
                            crossover_type=crossover_type,
                            target_bitwidth=3.0,
                            quant_weights_path=str(self.root),
                        ),
                        offspring=copy.deepcopy(base),
                        model=self.model,
                        grouped_layer_names=self.names,
                        target_cost_bits=self.cost(base)["total_cost_bits"],
                        budget_cost_kwargs=self.cost_kwargs,
                    )
                    exec(compile(block, search.__file__, "exec"), namespace)
                    repair.assert_called_once()
                    self.assertTrue(
                        repair.call_args.kwargs["preserve_equal_size_group_costs"]
                    )

    def test_generation_diagnostics_roundtrip_through_existing_csv_schema(self):
        base, donor = self.candidate(quant=(2, 4, 3, 3)), self.candidate(
            quant=(4, 2, 3, 3)
        )
        child, details = self.propose(base, donor)
        diagnostics = search.new_exchange_diagnostics()
        search.record_accepted_exchange(diagnostics, details, child, base, donor)
        summary = search.summarize_exchange_diagnostics(diagnostics)
        reporter = search.RunReporter(self.root / "report", "joint_depth_quant")
        reporter.append_generation(
            {"generation": 1, "mutation_summary": {"crossover_diagnostics": summary}}
        )
        with reporter.generation_log_path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(
            json.loads(row["mutation_summary"])["crossover_diagnostics"], summary
        )

    def test_checkpoint_defaults_roundtrip_and_seeded_resume(self):
        base, donor = self.candidate(quant=(2, 4, 3, 3)), self.candidate(
            quant=(4, 2, 3, 3)
        )
        child, details = self.propose(base, donor)
        diagnostics = search.new_exchange_diagnostics()
        diagnostics["crossover_no_legal_exchange"] = 5
        search.record_exchange_distance(diagnostics, "parent_distance", 2)
        search.record_accepted_exchange(diagnostics, details, child, base, donor)
        path = self.root / "checkpoint.pt"
        random.seed(89)
        save_search_checkpoint(
            path,
            search_type="joint_depth_quant",
            completed_generation=2,
            identity={"crossover_type": "local_exchange"},
            state={"exchange_diagnostics_total": diagnostics},
        )
        expected = self.propose(base, donor)
        checkpoint = load_search_checkpoint(path)
        restored = search.new_exchange_diagnostics(
            checkpoint["state"].get("exchange_diagnostics_total", {})
        )
        self.assertEqual(restored, diagnostics)
        restore_rng_state(checkpoint["rng_state"])
        self.assertEqual(self.propose(base, donor), expected)
        restored["parent_distance"]["count"] += 1
        self.assertEqual(diagnostics["parent_distance"]["count"], 1)
        # Execute the production resume assignment with a V2 state lacking keys.
        main_ast = ast.parse(inspect.getsource(search.main))
        resume = next(
            node
            for node in ast.walk(main_ast)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "exchange_diagnostics_total"
                for t in node.targets
            )
            and isinstance(node.value, ast.Call)
            and node.value.args
        )
        namespace = {
            "state": {},
            "new_exchange_diagnostics": search.new_exchange_diagnostics,
        }
        exec(
            compile(
                ast.Module(body=[resume], type_ignores=[]), search.__file__, "exec"
            ),
            namespace,
        )
        self.assertEqual(
            namespace["exchange_diagnostics_total"], search.new_exchange_diagnostics()
        )

    def test_cli_accepts_local_exchange_and_preserves_defaults(self):
        required = [
            "--model_name_or_path",
            "fixture",
            "--quant_weights_path",
            str(self.root),
            "--calibration_data",
            "wikitext2",
            "--drop_sparsity",
            "0.25",
            "--target_bitwidth",
            "3.0",
            "--generations",
            "20",
            "--offspring",
            "16",
            "--initially_generated",
            "32",
            "--initial_tokens",
            "512",
            "--survivors_per_selection",
            "8",
            "2",
            "1",
            "--tokens_per_selection",
            "512",
            "2048",
            "8192",
        ]
        defaults = search.parse_args(required)
        self.assertEqual(defaults.crossover_type, "component")
        self.assertEqual(defaults.crossover_parent_selection, "uniform")
        args = search.parse_args(
            required
            + [
                "--crossover_type",
                "local_exchange",
                "--population_size",
                "4",
                "--crossover_probability",
                "0.25",
            ]
        )
        search.validate_crossover_configuration(args)
        self.assertEqual(args.crossover_type, "local_exchange")


if __name__ == "__main__":
    unittest.main()
