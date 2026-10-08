"""Torch-free tests for the exact-budget replay planner (src/exact_replay.py)."""

import itertools
import json
import random
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import plan_exact_replays as planner  # noqa: E402
from scripts.build_depth_baseline_config import build_baseline_config  # noqa: E402
from src import exact_replay as er  # noqa: E402


class ExactReplayPureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = er.size_groups(er.mistral_module_sizes())
        cls.sources = planner.load_sources(REPO_ROOT, include_125=False)

    def test_groups_match_mistral_layout(self):
        self.assertEqual([len(g) for g in self.groups], [64, 64, 96])
        self.assertTrue(all(".k_proj" in n or ".v_proj" in n for n in self.groups[0]))
        self.assertTrue(all(".q_proj" in n or ".o_proj" in n for n in self.groups[1]))
        self.assertTrue(all(".mlp." in n for n in self.groups[2]))

    def test_required_level_sums(self):
        none = er.heuristic_mask("late_layer", 32, 0, 0)
        self.assertEqual(er.required_level_sums(self.groups, none), [192, 192, 288])
        s25 = er.heuristic_mask("late_layer", 32, 8, 8)
        self.assertEqual(er.required_level_sums(self.groups, s25), [196, 196, 294])
        s125 = er.heuristic_mask("late_layer", 32, 4, 4)
        self.assertEqual(er.required_level_sums(self.groups, s125), [194, 194, 291])

    def test_non_integer_group_budget_is_rejected(self):
        odd = er.heuristic_mask("late_layer", 32, 1, 1)
        with self.assertRaises(ValueError):
            er.required_level_sums(self.groups, odd)

    def test_all_completed_fullspace_finals_satisfy_group_budget(self):
        for label in planner.FULLSPACE_CANDIDATES:
            cand = self.sources[label]
            er.validate_group_budget(self.groups, cand["bits"], cand["drop"])

    def test_heuristic_masks_reproduce_early_baseline_rules(self):
        reference = ["attn+mlp"] * 8 + ["none"] * 24
        for method in ("random", "late_layer", "early_layer"):
            for seed in (0, 1, 7):
                config = build_baseline_config(reference, method, seed)
                mask = er.heuristic_mask(method, 32, 8, 8, seed)
                self.assertEqual(mask["attn"], [c in ("attn", "attn+mlp") for c in config])
                self.assertEqual(mask["mlp"], [c in ("mlp", "attn+mlp") for c in config])

    def test_late_layer_keep_last_keeps_final_layer(self):
        mask = er.heuristic_mask("late_layer_keep_last", 32, 8, 8)
        self.assertFalse(mask["attn"][31])
        self.assertEqual([i for i, v in enumerate(mask["attn"]) if v], list(range(23, 31)))

    def test_near_uniform_bits_are_feasible_and_even(self):
        for k in (4, 8):
            for rule, seed in (("late_layer", 0), ("random", 3), ("late_layer_keep_last", 0)):
                drop = er.heuristic_mask(rule, 32, k, k, seed)
                for order in ("spread", "random"):
                    bits = er.near_uniform_bits(self.groups, drop, order=order, seed=5)
                    er.validate_group_budget(self.groups, bits, drop)
                    for group in self.groups:
                        active = [bits[n] for n in group if er.module_is_active(n, drop)]
                        self.assertLessEqual(max(active) - min(active), 1)
        drop = er.heuristic_mask("late_layer", 32, 8, 8)
        bits = er.near_uniform_bits(self.groups, drop)
        active_mlp = [bits[n] for n in self.groups[2] if er.module_is_active(n, drop)]
        self.assertEqual(sorted(set(active_mlp)), [4, 5])
        self.assertEqual(active_mlp.count(5), 6)

    def test_score_mask_removes_lowest_scores(self):
        scores = {"attn": [float(i) for i in range(32)], "mlp": [float(31 - i) for i in range(32)]}
        mask = er.score_mask(scores, 3, 2)
        self.assertEqual([i for i, v in enumerate(mask["attn"]) if v], [0, 1, 2])
        self.assertEqual([i for i, v in enumerate(mask["mlp"]) if v], [30, 31])

    def test_owner_fill_exchanges_only_shared_genes(self):
        a, b = self.sources["E3s1"], self.sources["DWs0"]
        bits = er.fill_bits(a, b, a["drop"], self.groups, "owner")
        for group in self.groups:
            for name in group:
                shared = er.module_is_active(name, a["drop"]) and er.module_is_active(name, b["drop"])
                self.assertEqual(bits[name], b["bits"][name] if shared else a["bits"][name])
        self.assertEqual(er.level_deficits(self.groups, bits, a["drop"]), [0, 0, 0])

    def test_owner_deficits_of_a_pair_are_opposite(self):
        for a, b in (("E3s0", "E3s1"), ("J12s0", "J12s2"), ("IAs0", "P4s1")):
            ca, cb = self.sources[a], self.sources[b]
            dab = er.level_deficits(self.groups, er.fill_bits(ca, cb, ca["drop"], self.groups, "owner"), ca["drop"])
            dba = er.level_deficits(self.groups, er.fill_bits(cb, ca, cb["drop"], self.groups, "owner"), cb["drop"])
            self.assertEqual(dab, [-x for x in dba])

    def test_shared_gene_exchange_contrast_vanishes_for_additive_metric(self):
        rng = random.Random(0)
        weights = {n: rng.uniform(0.1, 2.0) for g in self.groups for n in g}
        removal = {k: [rng.uniform(0, 1) for _ in range(32)] for k in ("attn", "mlp")}

        def metric(drop, bits):
            value = sum(removal[k][i] for k in ("attn", "mlp") for i, v in enumerate(drop[k]) if v)
            value += sum(weights[n] / bits[n] for g in self.groups for n in g if er.module_is_active(n, drop))
            return value

        a, b = self.sources["J12s0"], self.sources["J12s1"]
        ab = er.fill_bits(a, b, a["drop"], self.groups, "owner")
        ba = er.fill_bits(b, a, b["drop"], self.groups, "owner")
        terms = er.interaction_terms(
            metric(a["drop"], a["bits"]), metric(b["drop"], b["bits"]),
            metric(a["drop"], ab), metric(b["drop"], ba),
        )
        self.assertAlmostEqual(terms["I"], 0.0, places=9)
        self.assertAlmostEqual(terms["I"], -(terms["delta_a"] + terms["delta_b"]), places=12)
        # The plain crossing does not have this property (stale genes differ).
        plain = er.interaction_terms(
            metric(a["drop"], a["bits"]), metric(b["drop"], b["bits"]),
            metric(a["drop"], b["bits"]), metric(b["drop"], a["bits"]),
        )
        self.assertNotAlmostEqual(plain["I"], 0.0, places=6)

    def test_eligibility_marks_only_exclusive_sublayers_active(self):
        a, b = self.sources["E3s0"], self.sources["E3s1"]
        elig = er.eligibility_drop_state(a["drop"], b["drop"])
        for kind in ("attn", "mlp"):
            for i in range(32):
                exclusive = (not a["drop"][kind][i]) and b["drop"][kind][i]
                self.assertEqual(not elig[kind][i], exclusive)

    def test_shift_keeps_levels_and_reduces_deficit(self):
        mask = self.sources["E3s0"]["drop"]
        e2 = self.sources["E2s0"]
        before = er.level_deficits(self.groups, e2["bits"], mask)
        shifted = er.shift_to_budget(e2["bits"], self.groups, mask)
        after = er.level_deficits(self.groups, shifted, mask)
        self.assertTrue(all(2 <= v <= 6 for v in shifted.values()))
        self.assertLess(sum(map(abs, after)), sum(map(abs, before)))
        self.assertLessEqual(sum(map(abs, after)), 3)

    def test_repair_capacity(self):
        drop = er.heuristic_mask("late_layer", 32, 8, 8)
        bits = {n: 6 for g in self.groups for n in g}
        self.assertEqual(er.repair_capacity(self.groups, bits, drop, [1, 0, -1]), [False, True, True])

    def test_planner_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            planner.main(["--output_dir", tmp])
            plan = json.loads((Path(tmp) / "replay_plan.json").read_text())
            ids = [job["id"] for job in plan["jobs"]]
            self.assertEqual(len(ids), len(set(ids)))
            free = {(j["pair"]["a"], j["pair"]["b"]) for j in plan["jobs"] if j["tier"] == "A"}
            self.assertEqual(free, {("E3s1", "DWs0"), ("J12s0", "J12s1")})
            for job in plan["jobs"]:
                deficits = job.get("static_deficits")
                if deficits is not None and any(deficits):
                    self.assertNotEqual(job["repair"]["scope"], "none", job["id"])
                if deficits is not None and not any(deficits) and "transform" not in job.get("bits", {}):
                    self.assertEqual(job["repair"]["scope"], "none", job["id"])
            analysis = (Path(tmp) / "pair_analysis.csv").read_text().splitlines()
            self.assertEqual(len(analysis), 1 + 48)

    def test_no_completed_pair_is_repair_free_under_plain_crossing(self):
        searched = [k for k in planner.FULLSPACE_CANDIDATES if planner.SPARSITY_OF[k] > 0]
        for a, b in itertools.combinations(searched, 2):
            if planner.SPARSITY_OF[a] != planner.SPARSITY_OF[b]:
                continue
            report = er.crossed_pair_report(a, self.sources[a], b, self.sources[b], self.groups)
            self.assertFalse(report["repair_free_pair"], (a, b))

    def test_metadata_constant(self):
        self.assertEqual(er.metadata_bits_per_weight(), Fraction(1, 4))

    def test_summarizer_computes_contrast_from_results(self):
        import csv as _csv
        from scripts import summarize_exact_replays as summ
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            planner.main(["--output_dir", tmp])
            plan = json.loads((out / "replay_plan.json").read_text())
            (out / "plan_used.json").write_text(json.dumps(plan))
            values = {
                "own_J12s0": 1.90, "own_J12s1": 1.95,
                "x_owner_exclusive_J12s0_QJ12s1_r0": 1.93,
                "x_owner_exclusive_J12s1_QJ12s0_r0": 1.97,
            }
            with (out / "replay_results.csv").open("w", newline="") as handle:
                writer = _csv.DictWriter(handle, fieldnames=["id", "status", "wikitext2_nll", "c4_nll",
                                                             "calibration_kl", "repair_changed_genes",
                                                             "wikitext2_ppl", "c4_ppl"])
                writer.writeheader()
                for job_id, nll in values.items():
                    writer.writerow({"id": job_id, "status": "completed", "wikitext2_nll": nll, "c4_nll": "",
                                     "calibration_kl": "", "repair_changed_genes": 0,
                                     "wikitext2_ppl": "", "c4_ppl": ""})
            summ.main(["--results_dir", tmp])
            rows = list(_csv.DictReader((out / "rq3_contrasts.csv").open()))
            row = next(r for r in rows if r["a"] == "J12s0" and r["fill"] == "owner")
            self.assertAlmostEqual(float(row["wikitext2_nll_I"]), (1.90 + 1.95) - (1.93 + 1.97))
            self.assertAlmostEqual(float(row["wikitext2_nll_delta_a"]), 0.03)
            self.assertAlmostEqual(float(row["wikitext2_nll_delta_b"]), 0.02)


if __name__ == "__main__":
    unittest.main()
