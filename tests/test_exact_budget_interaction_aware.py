"""Synthetic CPU tests for the explicit exact-budget mutation ablation."""

import ast
import copy
import inspect
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import evo_joint_search as search
from src.compression_budget import (
    CompressionBudgetError,
    candidate_compression_cost,
    repair_quant_state_to_budget,
    uniform_quantization_target_cost,
)


class ExactBudgetInteractionAwareTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = torch.nn.Module()
        self.model.model = torch.nn.Module()
        self.model.model.fixed = torch.nn.Parameter(torch.empty(8, device="meta"))
        self.model.model.layers = torch.nn.ModuleList()
        for _ in range(4):
            block = torch.nn.Module()
            for component, projections in (
                ("self_attn", {"q_proj": 2, "k_proj": 1, "v_proj": 1, "o_proj": 2}),
                ("mlp", {"gate_proj": 4, "up_proj": 4, "down_proj": 4}),
            ):
                container = torch.nn.Module()
                for name, out_features in projections.items():
                    module = torch.nn.Module()
                    module.weight = torch.nn.Parameter(
                        torch.empty((out_features, 32), device="meta")
                    )
                    container.add_module(name, module)
                block.add_module(component, container)
            self.model.model.layers.append(block)
        self.configure_model(num_layers=4, group_size=32)
        self.parent = self.make_parent()

    def configure_model(self, num_layers, group_size):
        names = [
            name
            for name, module in self.model.named_modules()
            if hasattr(module, "weight")
        ]
        groups = {}
        for name in names:
            groups.setdefault(self.model.get_submodule(name).weight.numel(), []).append(
                name
            )
        self.groups = [groups[size] for size in sorted(groups)]
        self.cost_kwargs = {
            "attention_module_names": [
                f"model.layers.{i}.self_attn" for i in range(num_layers)
            ],
            "mlp_module_names": [f"model.layers.{i}.mlp" for i in range(num_layers)],
            "dense_dtype_bits": 16,
            "group_size": group_size,
            "include_quantization_metadata": True,
            "scale_bits": 16,
            "zero_point_bits": 16,
        }
        self.target = uniform_quantization_target_cost(
            self.model, self.groups, 3, **self.cost_kwargs
        )["total_cost_bits"]
        for name in names:
            directory = self.root / name
            directory.mkdir(exist_ok=True)
            for level in range(2, 7):
                (directory / f"{level}.pth").touch()

    def make_parent(self, whole_block=False):
        num_layers = len(self.cost_kwargs["attention_module_names"])
        count = num_layers // 4
        drop = {
            "attn": [i < count for i in range(num_layers)],
            "mlp": [
                i < count if whole_block else count <= i < 2 * count
                for i in range(num_layers)
            ],
        }
        parent = {"drop": drop, "quant": [[3] * len(group) for group in self.groups]}
        parent["quant"] = self.repair(parent, rng=random.Random(77))
        return parent

    def cost(self, candidate):
        return candidate_compression_cost(
            self.model, candidate, grouped_layer_names=self.groups, **self.cost_kwargs
        )

    def repair(self, candidate, **kwargs):
        return repair_quant_state_to_budget(
            self.model,
            self.groups,
            str(self.root),
            candidate["quant"],
            candidate["drop"],
            self.target,
            preserve_equal_size_group_costs=True,
            uniform_reference_bitwidth=3,
            **self.cost_kwargs,
            **kwargs,
        )

    def propose(self, candidate=None, **kwargs):
        options = dict(
            target_cost_bits=self.target, budget_cost_kwargs=self.cost_kwargs
        )
        options.update(kwargs)
        with patch.object(
            search,
            "repair_active_quant_budget",
            side_effect=AssertionError("active repair in exact mode"),
        ):
            return search.mutate_interaction_aware_candidate(
                self.model,
                self.groups,
                str(self.root),
                candidate or self.parent,
                3.0,
                **options,
            )

    def cli(self, *extra):
        return search.parse_args(
            [
                "--model_name_or_path",
                "fixture",
                "--quant_weights_path",
                str(self.root),
                "--calibration_data",
                "wikitext2",
                "--drop_sparsity",
                "0.25",
                "--target_bitwidth",
                "3",
                "--generations",
                "1",
                "--offspring",
                "1",
                "--initially_generated",
                "1",
                "--initial_tokens",
                "512",
                "--survivors_per_selection",
                "1",
                "--tokens_per_selection",
                "512",
                "--group_rule",
                "size",
                *extra,
            ]
        )

    def exact_cli(self, *extra):
        return self.cli(
            "--compression_budget_mode",
            "match_uniform_quantization_total",
            "--budget_include_quantization_metadata",
            "--quantization_group_size",
            "32",
            *extra,
        )

    def test_parser_defaults_and_standard_exact_search_need_no_opt_in(self):
        defaults = self.cli()
        self.assertFalse(defaults.allow_exact_budget_ablation)
        self.assertEqual(defaults.compression_budget_mode, "legacy_active_average")
        self.assertEqual(defaults.joint_mutation_mode, "standard")
        self.assertEqual(defaults.crossover_type, "component")
        self.assertEqual(
            (defaults.population_size, defaults.crossover_probability), (1, 0.0)
        )
        args = self.exact_cli()
        search.validate_joint_search_args(args)
        # Programmatic callers with the old namespace retain the safe default.
        del args.allow_exact_budget_ablation
        search.validate_joint_search_args(args)

    def test_exact_interaction_requires_explicit_opt_in(self):
        args = self.exact_cli("--joint_mutation_mode", "interaction_aware")
        for has_flag in (True, False):
            with self.subTest(has_flag=has_flag):
                if not has_flag:
                    del args.allow_exact_budget_ablation
                with self.assertRaisesRegex(
                    ValueError, "--allow_exact_budget_ablation"
                ):
                    search.validate_joint_search_args(args)

    def test_exact_interaction_with_opt_in_and_legacy_active_mode_are_allowed(self):
        exact = self.exact_cli(
            "--joint_mutation_mode",
            "interaction_aware",
            "--allow_exact_budget_ablation",
        )
        self.assertFalse(exact.active_quant_budget)
        search.validate_joint_search_args(exact)
        active = self.cli(
            "--joint_mutation_mode", "interaction_aware", "--active_quant_budget"
        )
        search.validate_joint_search_args(active)

    def test_opt_in_does_not_bypass_other_reproduction_guards(self):
        for key, value, message in (
            ("active_quant_budget", True, "do not enable both"),
            ("group_rule", "none", "group_rule size"),
            ("quantization_group_size", None, "group_size is required"),
            ("target_bitwidth", 3.5, "integral"),
            ("joint_aware_mutation", True, "Joint-aware mutation"),
            ("sequential_mode", "depth_to_joint_warm", "Sequential initialization"),
        ):
            with self.subTest(key=key):
                args = self.exact_cli(
                    "--joint_mutation_mode",
                    "interaction_aware",
                    "--allow_exact_budget_ablation",
                )
                setattr(args, key, value)
                with self.assertRaisesRegex(ValueError, message):
                    search.validate_joint_search_args(args)
        args = self.cli(
            "--joint_mutation_mode",
            "interaction_aware",
            "--allow_exact_budget_ablation",
        )
        with self.assertRaisesRegex(ValueError, "requires --active_quant_budget"):
            search.validate_joint_search_args(args)

    def test_exact_cost_depth_counts_and_parent_immutability(self):
        for whole_block in (False, True):
            parent = self.make_parent(whole_block)
            before = copy.deepcopy(parent)
            for seed in range(5):
                with self.subTest(whole_block=whole_block, seed=seed):
                    random.seed(seed)
                    child, details = self.propose(parent, drop_entire_block=whole_block)
                    self.assertEqual(self.cost(child)["total_cost_bits"], self.target)
                    search.validate_depth_counts(child["drop"], 4, 1, whole_block)
                    self.assertNotEqual(child["drop"], parent["drop"])
                    self.assertTrue(details["preferred_quant_exchange_used"])
                    self.assertEqual(
                        details["depth_mask_entries_changed"], 4 if whole_block else 2
                    )
                    self.assertEqual(parent, before)
                    self.assertIsNot(child["quant"][0], parent["quant"][0])
                    self.assertIsNot(child["drop"]["attn"], parent["drop"]["attn"])

    def test_preferred_exchange_follows_repair_and_survives_generic_repair(self):
        exchanges = []
        real_mutate = search.mutate_quant_state

        def capture_exchange(model, groups, root, quant, step_size, drop, **kwargs):
            self.assertEqual(
                self.cost({"drop": drop, "quant": quant})["total_cost_bits"],
                self.target,
            )
            mutated = real_mutate(model, groups, root, quant, step_size, drop, **kwargs)
            exchanges.append(
                (copy.deepcopy(quant), mutated, copy.deepcopy(drop), kwargs)
            )
            return mutated

        random.seed(11)
        with patch.object(search, "mutate_quant_state", side_effect=capture_exchange):
            child, details = self.propose()
        self.assertEqual(len(exchanges), 1)
        before_quant, exchanged, drop, kwargs = exchanges[0]
        changed = [
            name
            for g, group in enumerate(self.groups)
            for i, name in enumerate(group)
            if before_quant[g][i] != exchanged[g][i]
        ]
        self.assertEqual(len(changed), 2)
        self.assertTrue(
            all(search.quant_layer_is_active(name, drop) for name in changed)
        )
        self.assertTrue(
            any(
                search.layer_index(name) in details["touched_layer_ids"]
                for name in changed
            )
        )
        self.assertEqual(
            kwargs["preferred_layer_ids"], set(details["touched_layer_ids"])
        )
        self.assertEqual(
            details["budget_repair_quant_changes"],
            search.count_quant_state_changes(self.parent["quant"], before_quant),
        )
        self.assertGreater(details["budget_repair_quant_changes"], 0)
        self.assertTrue(details["preferred_quant_exchange_used"])
        self.assertFalse(details["fallback_quant_exchange_used"])
        self.assertEqual(child["quant"], exchanged)
        rng_before = random.getstate()
        self.assertEqual(self.repair(child), exchanged)
        self.assertEqual(random.getstate(), rng_before)
        # Metadata depends on active shapes, not the exchanged bitwidths.
        before_cost = self.cost({"drop": drop, "quant": before_quant})
        self.assertEqual(
            self.cost(child)["quantization_metadata_bits"],
            before_cost["quantization_metadata_bits"],
        )

    def test_multiple_quant_exchanges_remain_exact_and_seeded(self):
        random.seed(29)
        first = self.propose(quant_mutations=3)
        random.seed(29)
        self.assertEqual(first, self.propose(quant_mutations=3))
        self.assertEqual(self.cost(first[0])["total_cost_bits"], self.target)
        self.assertEqual(self.repair(first[0]), first[0]["quant"])

    def test_fallback_exchange_still_obeys_exact_budget(self):
        real_mutate = search.mutate_quant_state

        def no_preferred(*args, **kwargs):
            if kwargs.get("preferred_layer_ids"):
                return copy.deepcopy(args[3])
            return real_mutate(*args, **kwargs)

        with patch.object(search, "mutate_quant_state", side_effect=no_preferred):
            child, details = self.propose()
        self.assertFalse(details["preferred_quant_exchange_used"])
        self.assertTrue(details["fallback_quant_exchange_used"])
        self.assertEqual(self.cost(child)["total_cost_bits"], self.target)

    def test_unrepresentable_budget_is_rejected(self):
        with self.assertRaises(CompressionBudgetError):
            self.propose(target_cost_bits=self.target + 1)

    def run_production_offspring_loop(self):
        """Exercise real dispatch/retries/budget checks without model evaluation."""
        main_ast = ast.parse(inspect.getsource(search.main)).body[0]
        generation = next(
            node
            for node in main_ast.body
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "generation"
        )
        start = next(
            i
            for i, node in enumerate(generation.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "offspring_list"
                for t in node.targets
            )
        )
        stop = next(
            i
            for i in range(start, len(generation.body))
            if isinstance(generation.body[i], ast.While)
        )
        args = self.exact_cli(
            "--joint_mutation_mode",
            "interaction_aware",
            "--allow_exact_budget_ablation",
        )
        args.max_offspring_attempts = 3
        namespace = dict(vars(search))
        namespace.update(
            args=args,
            model=self.model,
            grouped_layer_names=self.groups,
            population=[self.parent],
            crossover_enabled=False,
            exact_total_budget=True,
            target_cost_bits=self.target,
            budget_cost_kwargs=self.cost_kwargs,
            depth_mutation_limit=1,
            quant_mutation_count=1,
            exchange_diagnostics_total=search.new_exchange_diagnostics(),
        )
        namespace["mutation_offspring_accepted_total"] = 0
        exec(
            compile(
                ast.Module(body=generation.body[start : stop + 1], type_ignores=[]),
                search.__file__,
                "exec",
            ),
            namespace,
        )
        return namespace

    def test_offspring_loop_retries_failed_early_repair_and_preserves_exchange(self):
        repair_calls = []

        def fail_once(*args, **kwargs):
            before = copy.deepcopy(args[3])
            rng_before = random.getstate()
            if not repair_calls:
                repair_calls.append(None)
                raise CompressionBudgetError("synthetic infeasible depth proposal")
            repaired = repair_quant_state_to_budget(*args, **kwargs)
            repair_calls.append(
                (before, copy.deepcopy(repaired), rng_before, random.getstate())
            )
            return repaired

        random.seed(11)
        with patch.object(
            search, "repair_quant_state_to_budget", side_effect=fail_once
        ), patch.object(
            search,
            "repair_active_quant_budget",
            side_effect=AssertionError("active repair"),
        ):
            state = self.run_production_offspring_loop()
        self.assertEqual(state["infeasible_candidates"], 1)
        self.assertEqual(state["mutation_offspring_accepted_total"], 1)
        self.assertEqual(state["crossover_infeasible_candidates"], 0)
        self.assertEqual(state["offspring_attempts"], 2)
        self.assertEqual(
            len(repair_calls), 3
        )  # Failed early repair, successful early repair, generic no-op.
        before, after, rng_before, rng_after = repair_calls[-1]
        self.assertEqual(before, after)
        self.assertEqual(rng_before, rng_after)
        self.assertEqual(state["offspring_list"][0]["quant"], after)
        self.assertEqual(
            self.cost(state["offspring_list"][0])["total_cost_bits"], self.target
        )
        self.assertEqual(
            state["interaction_aware_totals"]["preferred_quant_exchanges_used"], 1
        )

    def test_mistral_shapes_match_the_reproduced_metadata_inclusive_target(self):
        # Reuse the existing shape-only fixture: no real weights or model loading.
        from test_compression_budget import Mistral7BV03Shape

        self.model = Mistral7BV03Shape()
        # Include only the 224 searched projections, not embeddings/head/norms.
        names = [
            name for name, _ in self.model.named_modules() if name.endswith("_proj")
        ]
        groups = {}
        for name in names:
            groups.setdefault(self.model.get_submodule(name).weight.numel(), []).append(
                name
            )
        self.groups = [groups[size] for size in sorted(groups)]
        self.cost_kwargs.update(
            attention_module_names=[f"model.layers.{i}.self_attn" for i in range(32)],
            mlp_module_names=[f"model.layers.{i}.mlp" for i in range(32)],
            group_size=128,
        )
        for name in names:
            directory = self.root / name
            directory.mkdir(exist_ok=True)
            for level in range(2, 7):
                (directory / f"{level}.pth").touch()
        self.target = uniform_quantization_target_cost(
            self.model, self.groups, 3, **self.cost_kwargs
        )["total_cost_bits"]
        self.assertEqual(len(names), 224)
        self.assertEqual(self.target, 26_982_023_168)
        parent = self.make_parent()
        random.seed(7)
        child, details = self.propose(parent)
        self.assertEqual(self.cost(child)["total_cost_bits"], 26_982_023_168)
        search.validate_depth_counts(child["drop"], 32, 8, False)
        self.assertTrue(details["preferred_quant_exchange_used"])
        self.assertEqual(self.repair(child), child["quant"])


if __name__ == "__main__":
    unittest.main()
