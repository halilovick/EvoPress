"""CPU-only coverage of the guarded depth warm-start and its production wiring."""

import ast
import copy
import inspect
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import evo_joint_search as search
from scripts.summarize_sequential_search import search_effort
from src.compression_budget import CompressionBudgetError
from src.run_reporting import RunReporter
from src.search_checkpoint import (
    load_search_checkpoint,
    save_search_checkpoint,
    validate_checkpoint_identity,
)
from src.sequential_search import SequentialSearchError
from test_compression_budget import Mistral7BV03Shape


def assignment_index(body, name):
    return next(
        i for i, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    )


def execute_between(body, start, stop, namespace):
    """Run actual main() blocks, bypassing GPU/model/data setup only."""
    nodes = body[assignment_index(body, start):assignment_index(body, stop)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), search.__file__, "exec"), namespace)


class ExactBudgetDepthWarmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.database_temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.database_temp.cleanup)
        cls.database = Path(cls.database_temp.name)
        cls.model = Mistral7BV03Shape()  # Meta tensors: shapes only, no model weights.
        by_size = {}
        for name, module in cls.model.named_modules():
            if name.endswith("_proj"):
                by_size.setdefault(module.weight.numel(), []).append(name)
                directory = cls.database / name
                directory.mkdir()
                for level in range(2, 7):
                    (directory / f"{level}.pth").touch()
        cls.groups = [by_size[size] for size in sorted(by_size)]
        cls.cost_kwargs = {
            "attention_module_names": [f"model.layers.{i}.self_attn" for i in range(32)],
            "mlp_module_names": [f"model.layers.{i}.mlp" for i in range(32)],
            "dense_dtype_bits": 16,
            "group_size": 128,
            "include_quantization_metadata": True,
            "scale_bits": 16,
            "zero_point_bits": 16,
        }
        cls.target = 26_982_023_168
        cls.main_body = ast.parse(inspect.getsource(search.main)).body[0].body
        cls.generation_body = next(
            node.body for node in cls.main_body
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name) and node.target.id == "generation"
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_dir = self.root / "depth_source"
        self.source_dir.mkdir()
        self.drop = {
            "attn": [i < 8 for i in range(32)],
            "mlp": [8 <= i < 16 for i in range(32)],
        }
        self.source_candidate = {
            "candidate_type": "depth_only",
            "attention_mask": [int(x) for x in self.drop["attn"]],
            "mlp_mask": [int(x) for x in self.drop["mlp"]],
            "candidate_vector_raw": copy.deepcopy(self.drop),
            "bitwidth_by_module": {},
        }
        self.source_summary = {
            "model_name": "mistralai/Mistral-7B-v0.3",
            "search_type": "depth_only",
            "dataset_calibration": "wikitext2",
            "depth_statistics": {"num_layers": 32},
            "compression_config": {"drop_entire_block": False},
            "search_config": {
                "seed": 7,
                "generations": 20,
                "offspring": 16,
                "candidate_evaluations_search_total": 572,
                "evaluation_tokens_search_total": 999424,
            },
            "final_metrics": {"runtime_seconds": 123.0},
        }
        self.write_source()

    def write_source(self):
        (self.source_dir / "final_candidate.json").write_text(
            json.dumps(self.source_candidate), encoding="utf-8"
        )
        (self.source_dir / "run_summary.json").write_text(
            json.dumps(self.source_summary), encoding="utf-8"
        )

    def cli(self, *extra):
        return search.parse_args([
            "--model_name_or_path", "mistralai/Mistral-7B-v0.3",
            "--quant_weights_path", str(self.database),
            "--output_dir", str(self.root / "joint"),
            "--target_bitwidth", "3", "--drop_sparsity", "0.25",
            "--calibration_data", "fineweb_edu",
            "--group_rule", "size", "--initially_generated", "1",
            "--initial_tokens", "2048", "--generations", "20",
            "--offspring", "128", "--survivors_per_selection", "16", "4", "1",
            "--tokens_per_selection", "2048", "16384", "131072",
            "--seed", "0", *extra,
        ])

    def exact_cli(self, *extra):
        return self.cli(
            "--compression_budget_mode", "match_uniform_quantization_total",
            "--quantization_group_size", "128",
            "--budget_include_quantization_metadata",
            "--expected_target_cost_bits", str(self.target), *extra,
        )

    def warm_cli(self, *extra):
        return self.exact_cli(
            "--sequential_mode", "depth_to_joint_warm",
            "--stage1_run_dir", str(self.source_dir),
            "--allow_exact_budget_ablation",
            "--skip_initial_single_candidate_evaluation", *extra,
        )

    def namespace(self, args):
        search.validate_joint_search_args(args)
        artifacts = imported = None
        if args.sequential_mode != "none":
            artifacts = search.resolve_stage1_artifacts(args.stage1_run_dir, args.stage1_candidate)
            imported = search.load_stage1_depth_candidate(
                artifacts, expected_model_name=args.model_name_or_path,
                num_layers=32, drop_count=8, drop_entire_block=False,
            )
        namespace = dict(vars(search))
        namespace.update(
            args=args, model=self.model, grouped_layer_names=self.groups,
            total_blocks=32, blocks_to_remove=8, layers=[],
            calibration_data=[], target_logits=[],
            stage1_artifacts=artifacts, stage1_import=imported,
            exact_total_budget=args.compression_budget_mode == "match_uniform_quantization_total",
            exact_depth_warm_ablation=search.is_exact_budget_depth_warm_ablation(args),
            target_cost_bits=self.target, budget_cost_kwargs=self.cost_kwargs,
            effective_selection_survivors=search.effective_survivors_per_selection(
                args.survivors_per_selection, args.population_size
            ),
        )
        return namespace

    def initialize(self, args=None):
        state = self.namespace(args or self.warm_cli())
        execute_between(
            self.main_body, "initial_candidates", "initial_fixed_quant_legal_swap_count", state
        )
        return state

    def cost(self, candidate):
        return search.candidate_compression_cost(
            self.model, candidate, grouped_layer_names=self.groups, **self.cost_kwargs
        )

    def identity(self, state):
        execute_between(self.main_body, "checkpoint_identity", "checkpoint_path", state)
        return state["checkpoint_identity"]

    def warm_summary(self, state):
        execute_between(self.main_body, "exact_depth_warm_summary", "crossover_summary", state)
        return state["exact_depth_warm_summary"]

    def test_parser_defaults_and_standard_exact_search_without_opt_in(self):
        args = self.cli()
        self.assertFalse(args.allow_exact_budget_ablation)
        self.assertEqual(args.sequential_mode, "none")
        self.assertEqual(args.compression_budget_mode, "legacy_active_average")
        self.assertEqual(args.joint_mutation_mode, "standard")
        self.assertEqual((args.population_size, args.crossover_probability), (1, 0.0))
        self.assertFalse(args.skip_initial_single_candidate_evaluation)
        standard = self.exact_cli("--skip_initial_single_candidate_evaluation")
        search.validate_joint_search_args(standard)
        del standard.allow_exact_budget_ablation
        search.validate_joint_search_args(standard)
        random.seed(0)
        self.assertEqual(self.cost(self.initialize(standard)["parent"])["total_cost_bits"], self.target)

    def test_generated_paper_command_still_validates_without_opt_in(self):
        from test_apples_to_apples_workflow import PAPER_CONFIG
        from scripts.run_apples_to_apples import build_command, read_json

        command = build_command(
            "joint", read_json(PAPER_CONFIG), 0, self.database,
            self.root / "paper", "python", "torchrun", None,
        )
        args = search.parse_args(command[2:])
        search.validate_joint_search_args(args)
        self.assertFalse(args.allow_exact_budget_ablation)
        self.assertEqual(args.sequential_mode, "none")
        self.assertTrue(args.skip_initial_single_candidate_evaluation)

    def test_warm_requires_opt_in_with_or_without_initial_evaluation(self):
        for skip in (False, True):
            for missing_flag_attribute in (False, True):
                with self.subTest(skip=skip, missing=missing_flag_attribute):
                    args = self.warm_cli()
                    args.skip_initial_single_candidate_evaluation = skip
                    args.allow_exact_budget_ablation = False
                    if missing_flag_attribute:
                        del args.allow_exact_budget_ablation
                    with self.assertRaisesRegex(ValueError, "--allow_exact_budget_ablation"):
                        search.validate_joint_search_args(args)

    def test_explicit_warm_accepts_both_initial_evaluation_policies(self):
        for skip in (False, True):
            args = self.warm_cli()
            args.skip_initial_single_candidate_evaluation = skip
            search.validate_joint_search_args(args)
            self.assertTrue(search.is_exact_budget_depth_warm_ablation(args))

    def test_other_sequential_modes_and_combined_interaction_remain_rejected(self):
        for mode in ("depth_to_quant_frozen", "quant_to_depth_frozen", "quant_to_joint_warm"):
            with self.subTest(mode=mode):
                args = self.warm_cli()
                args.sequential_mode = mode
                args.skip_initial_single_candidate_evaluation = False
                with self.assertRaisesRegex(ValueError, "Sequential initialization"):
                    search.validate_joint_search_args(args)
        args = self.warm_cli("--joint_mutation_mode", "interaction_aware")
        args.skip_initial_single_candidate_evaluation = False
        with self.assertRaisesRegex(ValueError, "Sequential initialization"):
            search.validate_joint_search_args(args)

    def test_warm_does_not_bypass_other_guards(self):
        for key, value in (
            ("population_size", 2), ("crossover_probability", 0.5),
            ("initially_generated", 2), ("active_quant_budget", True),
            ("group_rule", "none"), ("target_bitwidth", 3.5),
            ("quantization_group_size", None), ("joint_aware_mutation", True),
            ("adaptive_mutation", True), ("coarse_to_fine_mutation", True),
        ):
            with self.subTest(key=key):
                args = self.warm_cli()
                args.skip_initial_single_candidate_evaluation = False
                setattr(args, key, value)
                with self.assertRaises(ValueError):
                    search.validate_joint_search_args(args)

    def test_active_budget_warm_initialization_is_unchanged(self):
        args = self.cli(
            "--sequential_mode", "depth_to_joint_warm",
            "--stage1_run_dir", str(self.source_dir), "--active_quant_budget",
        )
        with patch.object(search, "repair_quant_state_to_budget", side_effect=AssertionError("exact repair")), patch.object(
            search, "repair_active_quant_budget", wraps=search.repair_active_quant_budget
        ) as active_repair, patch.object(search, "selection", return_value=([{
            "drop": copy.deepcopy(self.drop), "quant": [[3] * len(g) for g in self.groups]
        }], [0.25])):
            state = self.initialize(args)
        active_repair.assert_called_once()
        self.assertFalse(state["exact_depth_warm_ablation"])
        self.assertEqual(state["parent"]["drop"], self.drop)
        self.assertEqual(state["parent"]["quant"], [[3] * len(g) for g in self.groups])
        self.assertEqual(state["initial_candidate_evaluations"], 1)
        self.assertEqual(state["initial_evaluation_tokens"], 2048)
        self.assertEqual(self.warm_summary(state), {})
        args.skip_initial_single_candidate_evaluation = True
        with self.assertRaises(ValueError):
            search.validate_joint_search_args(args)

    def test_initialization_preserves_import_and_matches_baseline_exact_repair(self):
        # Only the depth component is imported, even if the source carries
        # quantization metadata. Fresh genes must still start at reference 3.
        self.source_candidate["bitwidth_by_module"] = {
            name: 6 for group in self.groups for name in group
        }
        self.write_source()
        source_bytes = (self.source_dir / "final_candidate.json").read_bytes()
        random.seed(0)
        with patch.object(search, "repair_active_quant_budget", side_effect=AssertionError("active repair")), patch.object(
            search, "repair_quant_state_to_budget", wraps=search.repair_quant_state_to_budget
        ) as repair:
            state = self.initialize()
        repair.assert_called_once()
        self.assertEqual(repair.call_args.args[3], [[3] * len(g) for g in self.groups])
        self.assertEqual(repair.call_args.args[4], self.drop)
        self.assertEqual(repair.call_args.kwargs, {
            "preserve_equal_size_group_costs": True,
            "uniform_reference_bitwidth": 3, **self.cost_kwargs,
        })
        parent = state["parent"]
        search.validate_depth_counts(parent["drop"], 32, 8, False)
        self.assertEqual(parent["drop"], self.drop)
        self.assertEqual(state["stage1_import"].component, self.drop)
        self.assertIsNot(parent["drop"]["attn"], state["stage1_import"].component["attn"])
        self.assertEqual((self.source_dir / "final_candidate.json").read_bytes(), source_bytes)
        self.assertEqual(state["stage1_artifacts"].candidate, self.source_candidate)
        self.assertEqual(sum(map(len, parent["quant"])), 224)
        self.assertTrue(all(2 <= bit <= 6 for group in parent["quant"] for bit in group))
        self.assertGreater(len(state["initial_repair_changed_gene_names"]), 0)
        cost = self.cost(parent)
        self.assertEqual(cost["total_cost_bits"], self.target)
        self.assertEqual(cost["total_cost_bits"] - self.target, 0)
        self.assertEqual(cost["dense_model_bits"], 115_968_376_832)
        self.assertEqual(cost["quantization_metadata_bits"], 1_308_622_848)
        self.assertAlmostEqual(cost["assigned_average_bitwidth_active"], 49 / 12)

        # With the same mask and RNG, the unchanged baseline branch must agree.
        random.seed(0)
        with patch.object(search, "make_random_drop_state", return_value=copy.deepcopy(self.drop)):
            baseline = self.initialize(self.exact_cli("--skip_initial_single_candidate_evaluation"))
        self.assertEqual(baseline["parent"], parent)
        self.assertEqual(self.cost(baseline["parent"]), cost)
        self.assertEqual((state["initial_candidate_evaluations"], state["initial_evaluation_tokens"]), (0, 0))

    def test_bad_depth_counts_are_rejected_before_initialization(self):
        self.source_candidate["attention_mask"][0] = 0
        self.source_candidate["candidate_vector_raw"]["attn"][0] = False
        self.write_source()
        with self.assertRaises(SequentialSearchError):
            self.initialize()

    def test_inexact_target_fails_without_changing_import(self):
        state = self.namespace(self.warm_cli())
        state["target_cost_bits"] += 1
        with self.assertRaises(CompressionBudgetError):
            execute_between(self.main_body, "initial_candidates", "initial_fixed_quant_legal_swap_count", state)
        self.assertEqual(state["stage1_import"].component, self.drop)

    def test_fixed_seed_initialization_is_deterministic(self):
        random.seed(19)
        first = self.initialize(self.warm_cli("--offspring", "1"))
        first_child = self.offspring(first)
        random.seed(19)
        second = self.initialize(self.warm_cli("--offspring", "1"))
        second_child = self.offspring(second)
        self.assertEqual(first["parent"], second["parent"])
        self.assertEqual(first["initial_repair_changed_gene_names"], second["initial_repair_changed_gene_names"])
        self.assertEqual(first_child, second_child)
        self.assertEqual(first["offspring_mutation_types"], second["offspring_mutation_types"])

    def offspring(self, state):
        state.update(
            crossover_enabled=False, depth_mutation_limit=3, quant_mutation_count=1,
            exchange_diagnostics_total=search.new_exchange_diagnostics(),
            mutation_offspring_accepted_total=0,
        )
        execute_between(self.generation_body, "offspring_list", "stage_candidate_evaluations", state)
        return state["offspring_list"][0]

    def test_generation_one_uses_standard_mutation_and_neither_component_is_frozen(self):
        for probability, mutation_type in ((0.0, "depth"), (0.99, "quantization")):
            with self.subTest(mutation=mutation_type):
                args = self.warm_cli("--offspring", "1")
                random.seed(0)
                state = self.initialize(args)
                before = copy.deepcopy(state["parent"])
                # Functions in the namespace are the actual production globals.
                with patch.object(search.random, "random", return_value=probability):
                    for name in ("mutate_interaction_aware_candidate", "mutate_fixed_quant_depth_candidate", "repair_active_quant_budget", "validate_frozen_component"):
                        state[name] = Mock(side_effect=AssertionError(name))
                    state["mutate_drop_state"] = Mock(wraps=search.mutate_drop_state)
                    state["mutate_quant_state"] = Mock(wraps=search.mutate_quant_state)
                    child = self.offspring(state)
                self.assertEqual(state["offspring_mutation_types"], [mutation_type])
                self.assertEqual(self.cost(child)["total_cost_bits"], self.target)
                search.validate_depth_counts(child["drop"], 32, 8, False)
                if mutation_type == "depth":
                    self.assertNotEqual(child["drop"], before["drop"])
                    state["mutate_drop_state"].assert_called()
                    state["mutate_quant_state"].assert_not_called()
                else:
                    self.assertNotEqual(child["quant"], before["quant"])
                    self.assertEqual(child["drop"], before["drop"])
                    state["mutate_quant_state"].assert_called()
                    state["mutate_drop_state"].assert_not_called()
                    self.assertIsNone(state["mutate_quant_state"].call_args.args[5])
                self.assertEqual(state["parent"], before)
                self.assertEqual(state["stage1_artifacts"].candidate, self.source_candidate)

    def test_g20_production_selection_counters_exclude_stage_one(self):
        random.seed(0)
        state = self.initialize()
        parent = state["parent"]
        # Distinct valid candidates differ only in inactive assignments, avoiding
        # expensive evaluation while exercising real elitism and counter code.
        inactive = [(g, i) for g, group in enumerate(self.groups) for i, name in enumerate(group)
                    if not search.quant_layer_is_active(name, parent["drop"])]
        candidates = []
        for index in range(128):
            child = copy.deepcopy(parent)
            g, i = inactive[0]
            child["quant"][g][i] = 2
            for digit in range(4):
                g, i = inactive[digit + 1]
                child["quant"][g][i] = 2 + (index // (5 ** digit)) % 5
            candidates.append(child)

        def selection(**kwargs):
            pool = kwargs["candidates"]
            chosen = [parent] if kwargs["num_survive"] == 1 else pool[:kwargs["num_survive"]]
            return chosen, [0.0] * len(chosen)

        state.update(
            selection=Mock(side_effect=selection), generation_population=[parent],
            offspring_attempts=128, offspring_attempts_total=0,
            candidate_evaluations_search_cumulative=state["initial_candidate_evaluations"],
            evaluation_tokens_search_cumulative=state["initial_evaluation_tokens"],
        )
        for generation in range(20):
            state.update(generation=generation, offspring_list=list(candidates),
                         offspring_mutation_types=["quantization"] * 128)
            execute_between(self.generation_body, "stage_candidate_evaluations", "population", state)
        self.assertEqual(state["candidate_evaluations_search_cumulative"], 2980)
        self.assertEqual(state["evaluation_tokens_search_cumulative"], 23_592_960)
        self.assertEqual(state["selection"].call_count, 60)

    def test_checkpoint_records_source_hash_and_rejects_replaced_mask(self):
        random.seed(0)
        state = self.initialize()
        identity = self.identity(state)
        self.assertEqual(identity["exact_budget_ablation"], "depth_to_joint_warm")
        self.assertTrue(identity["allow_exact_budget_ablation"])
        self.assertTrue(identity["skip_initial_single_candidate_evaluation"])
        self.assertEqual(identity["stage1_candidate_hash"], search.stable_json_hash(self.drop))
        checkpoint_path = self.root / "checkpoint.pt"
        save_search_checkpoint(checkpoint_path, search_type="joint_depth_quant", completed_generation=1,
                               identity=identity, state={"initial_parent": state["initial_parent"]})
        checkpoint = load_search_checkpoint(checkpoint_path)
        validate_checkpoint_identity(checkpoint, self.identity(self.namespace(self.warm_cli())))
        # Replace the source at exactly the same path, preserving legal counts.
        for mask in (self.source_candidate["attention_mask"], self.source_candidate["candidate_vector_raw"]["attn"]):
            mask[0], mask[31] = mask[31], mask[0]
        self.write_source()
        with self.assertRaisesRegex(ValueError, "stage1_candidate_hash"):
            validate_checkpoint_identity(checkpoint, self.identity(self.namespace(self.warm_cli())))

    def test_old_checkpoint_identities_are_unchanged(self):
        for args in (self.exact_cli(), self.cli("--sequential_mode", "depth_to_joint_warm",
                    "--stage1_run_dir", str(self.source_dir), "--active_quant_budget")):
            with self.subTest(mode=args.compression_budget_mode):
                old_identity = self.identity(self.namespace(args))
                self.assertTrue({"exact_budget_ablation", "allow_exact_budget_ablation",
                                 "stage1_candidate_hash", "skip_initial_single_candidate_evaluation"}.isdisjoint(old_identity))
                args.allow_exact_budget_ablation = True
                current = self.identity(self.namespace(args))
                validate_checkpoint_identity({"identity": old_identity}, current)
                self.assertEqual(current, old_identity)

    def test_reporting_preserves_source_metadata_without_folding_in_compute(self):
        random.seed(0)
        state = self.initialize()
        metadata = self.warm_summary(state)
        self.assertEqual(metadata["initial_compression_cost_bits"], self.target)
        self.assertEqual(metadata["initial_compression_difference_bits"], 0)
        self.assertTrue(metadata["initial_evaluation_skipped"])
        self.assertFalse(metadata["stage1_compute_included_in_search_totals"])
        self.assertEqual(metadata["stage1_source_metadata"], self.source_summary)
        self.assertIsNot(metadata["stage1_source_metadata"], state["stage1_artifacts"].summary)
        self.assertEqual(Path(metadata["stage1_summary_path"]), self.source_dir / "run_summary.json")
        sequential = search.build_sequential_summary_metadata(
            mode=state["args"].sequential_mode, stage1_import=state["stage1_import"],
            quant_initialization_policy="strict",
            initial_repair_changed_gene_names=state["initial_repair_changed_gene_names"],
            initial_candidate_count=1, initial_parent=state["initial_parent"],
            final_parent=state["parent"], initial_fixed_quant_legal_swap_count=None,
            final_fixed_quant_legal_swap_count=None, active_budget_valid=None, depth_counts_valid=True,
        )
        self.assertIsNone(sequential["frozen_component"])
        # Evaluate the actual extra_summary expression passed to the reporter.
        report = next(node.value for node in self.main_body if isinstance(node, ast.Expr)
                      and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                      and node.value.func.attr == "write_summary")
        expression = next(kw.value for kw in report.keywords if kw.arg == "extra_summary")
        state.update(sequential_summary=sequential, crossover_summary={})
        extras = eval(compile(ast.Expression(expression), search.__file__, "eval"), state)
        state.update(
            eval_tokens_by_dataset={}, resume_source_checkpoint=None,
            resumed_from_generation=0, crossover_enabled=False,
            offspring_attempts_total=2560,
            candidate_evaluations_search_cumulative=2980,
            evaluation_tokens_search_cumulative=23_592_960,
        )
        expression = next(kw.value for kw in report.keywords if kw.arg == "search_config")
        search_config = eval(compile(ast.Expression(expression), search.__file__, "eval"), state)
        with patch("src.run_reporting.peak_gpu_memory", return_value=(None, None)):
            reporter = RunReporter(self.root / "report", search_type="joint_depth_quant")
            path = reporter.write_summary(
                model_name=state["args"].model_name_or_path, dataset_calibration="fineweb_edu",
                dataset_eval=[], search_config=search_config,
                compression_config={}, final_metrics={}, parameter_statistics={}, depth_statistics={},
                quantization_statistics={}, model_size_statistics={}, artifacts={}, extra_summary=extras,
            )
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(summary["exact_budget_ablation"], "depth_to_joint_warm")
        self.assertTrue(summary["search_config"]["allow_exact_budget_ablation"])
        self.assertTrue(summary["search_config"]["skip_initial_single_candidate_evaluation"])
        self.assertEqual(summary["search_config"]["sequential_mode"], "depth_to_joint_warm")
        self.assertEqual(summary["search_config"]["initial_candidate_evaluations"], 0)
        self.assertEqual(summary["search_config"]["initial_evaluation_tokens"], 0)
        self.assertEqual(summary["stage1_source_metadata"]["search_config"]["seed"], 7)
        self.assertEqual(search_effort(summary["search_config"]), {
            "candidate_evaluations": 2980, "evaluated_tokens": 23_592_960,
        })

    def test_missing_source_metadata_is_not_invented(self):
        (self.source_dir / "run_summary.json").unlink()
        random.seed(0)
        state = self.initialize()
        metadata = self.warm_summary(state)
        self.assertIsNone(metadata["stage1_source_metadata"])
        self.assertIsNone(metadata["stage1_summary_path"])
        self.assertEqual(metadata["initial_compression_cost_bits"], self.target)


class SequentialEffortTest(unittest.TestCase):
    def legacy(self):
        return {"initial_candidates": 1, "initial_candidates_evaluated": 1,
                "initial_tokens": 2048, "generations": 20, "offspring": 128,
                "selection_survivors": [16, 4, 1], "selection_tokens": [2048, 16384, 131072]}

    def test_legacy_reconstruction_unchanged(self):
        self.assertEqual(search_effort(self.legacy()), {
            "candidate_evaluations": 2981, "evaluated_tokens": 23_595_008,
        })

    def test_initial_evaluation_counters_override_legacy_candidate_count(self):
        search_config = self.legacy()
        search_config.update(initial_candidate_evaluations=0, initial_evaluation_tokens=0)
        self.assertEqual(search_effort(search_config), {
            "candidate_evaluations": 2980, "evaluated_tokens": 23_592_960,
        })

    def test_recorded_totals_override_reconstruction_independently(self):
        search_config = self.legacy()
        search_config["candidate_evaluations_search_total"] = 149
        self.assertEqual(search_effort(search_config)["candidate_evaluations"], 149)
        self.assertEqual(search_effort(search_config)["evaluated_tokens"], 23_595_008)
        del search_config["candidate_evaluations_search_total"]
        search_config["evaluation_tokens_search_total"] = 1179648
        self.assertEqual(search_effort(search_config)["candidate_evaluations"], 2981)
        self.assertEqual(search_effort(search_config)["evaluated_tokens"], 1179648)
        self.assertEqual(search_effort({"candidate_evaluations_search_total": 0,
                                     "evaluation_tokens_search_total": 0}), {
            "candidate_evaluations": 0, "evaluated_tokens": 0,
        })


if __name__ == "__main__":
    unittest.main()
