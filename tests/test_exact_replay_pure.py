"""Torch-free tests for the exact-budget replay planner (src/exact_replay.py)."""

import itertools
import json
import math
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

    def test_random_fill_only_changes_shared_genes_and_matches_deficit(self):
        a, b = self.sources["E3s0"], self.sources["E3s1"]
        real_deficits = er.level_deficits(
            self.groups, er.fill_bits(a, b, a["drop"], self.groups, "owner"), a["drop"]
        )
        self.assertTrue(any(real_deficits))  # sanity: this pair actually needs repair
        bits = er.random_fill_bits(a, b["drop"], a["drop"], self.groups, real_deficits, seed=0)
        for group in self.groups:
            for name in group:
                shared = er.module_is_active(name, a["drop"]) and er.module_is_active(name, b["drop"])
                if not shared:
                    self.assertEqual(bits[name], a["bits"][name])
        # The whole point of the control: it reproduces the exact real deficit.
        self.assertEqual(er.level_deficits(self.groups, bits, a["drop"]), real_deficits)

    def test_random_fill_is_seed_deterministic_and_varies_with_seed(self):
        a, b = self.sources["J12s0"], self.sources["J12s2"]
        deficits = er.level_deficits(
            self.groups, er.fill_bits(a, b, a["drop"], self.groups, "owner"), a["drop"]
        )
        bits0a = er.random_fill_bits(a, b["drop"], a["drop"], self.groups, deficits, seed=0)
        bits0b = er.random_fill_bits(a, b["drop"], a["drop"], self.groups, deficits, seed=0)
        bits1 = er.random_fill_bits(a, b["drop"], a["drop"], self.groups, deficits, seed=1)
        self.assertEqual(bits0a, bits0b)
        self.assertNotEqual(bits0a, bits1)
        # Both seeds must still reproduce the same target deficit.
        self.assertEqual(er.level_deficits(self.groups, bits1, a["drop"]), deficits)

    def test_random_fill_round_trips_through_build_replay_bits(self):
        # Same (mask, repair-scope) contract as a real Tier B job: build_replay_bits
        # must hand back bits plus the identical eligibility mask that a real
        # x_owner_exclusive_{a}_Q{b} job would use.
        a, b = self.sources["E3s1"], self.sources["E3s2"]
        deficits = er.level_deficits(
            self.groups, er.fill_bits(a, b, a["drop"], self.groups, "owner"), a["drop"]
        )
        job_entry = {
            "mask": {"source": "E3s1"},
            "bits": {"rule": "random_fill", "source": "E3s1", "eligible_from": "E3s2",
                     "deficits": deficits, "seed": 2},
            "repair": {"scope": "exclusive", "seed": 2},
        }
        bits, eligibility = er.build_replay_bits(job_entry, a["drop"], self.sources, self.groups)
        expected_elig = er.eligibility_drop_state(a["drop"], b["drop"])
        self.assertEqual(eligibility, expected_elig)
        direct = er.random_fill_bits(a, b["drop"], a["drop"], self.groups, deficits, seed=2)
        self.assertEqual(bits, direct)
        # And build_replay_bits's result must be the exact real deficit, not
        # merely "some" deficit -- this is what makes it a matched control.
        self.assertEqual(er.level_deficits(self.groups, bits, a["drop"]), deficits)

    def test_planner_adds_matched_bctrl_jobs_for_every_tier_b_pair(self):
        # Tier B's own `pair` field only records the canonical (a, b) order a
        # pair was analysed in, not which of the two jobs in that call is the
        # mask owner -- so directions are read off the job ids, which do
        # encode direction for both tiers (`x_owner_exclusive_{mask}_Q{other}`
        # and `rnd_{mask}_vs_{other}`).
        with tempfile.TemporaryDirectory() as tmp:
            planner.main(["--output_dir", tmp])
            plan = json.loads((Path(tmp) / "replay_plan.json").read_text())

        def directions(ids, prefix, infix):
            out = set()
            for job_id in ids:
                if not job_id.startswith(prefix):
                    continue
                body = job_id[len(prefix):].rsplit("_r", 1)[0]
                mask, other = body.split(infix)
                out.add((mask, other))
            return out

        all_ids = [j["id"] for j in plan["jobs"]]
        b_directions = directions([j["id"] for j in plan["jobs"] if j["tier"] == "B"], "x_owner_exclusive_", "_Q")
        ctrl_directions = directions([j["id"] for j in plan["jobs"] if j["tier"] == "B-ctrl"], "rnd_", "_vs_")
        self.assertEqual(b_directions, ctrl_directions)
        self.assertTrue(b_directions)  # sanity: not vacuously equal
        for mask, other in ctrl_directions:
            ctrl_ids = [i for i in all_ids if i.startswith(f"rnd_{mask}_vs_{other}_r")]
            self.assertEqual(len(ctrl_ids), len(planner.REPAIR_SEEDS))

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

    def test_analyze_null_control_computes_excess_from_results(self):
        import csv as _csv
        from scripts import analyze_null_control as nullctrl
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            # math.log(PPL) values chosen so the arithmetic is easy to check by hand:
            # own=2.00; real crossed mean nll=2.10 (delta_real=+0.10);
            # null mean nll=2.04 (delta_null=+0.04); excess = +0.06.
            rows = {
                "own_E3s0": {"wikitext2_ppl": math.exp(2.00), "repair_changed_genes": 0},
                "x_owner_exclusive_E3s0_QE3s1_r0": {"wikitext2_ppl": math.exp(2.10), "repair_changed_genes": 3},
                "rnd_E3s0_vs_E3s1_r0": {"wikitext2_ppl": math.exp(2.04), "repair_changed_genes": 5},
            }
            fieldnames = ["id", "status", "wikitext2_ppl", "calibration_kl", "repair_changed_genes"]
            with (out / "replay_results.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = _csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                for job_id, values in rows.items():
                    writer.writerow({"id": job_id, "status": "completed", "calibration_kl": "", **values})
            nullctrl.main(["--results_dir", str(out)])
            records = list(_csv.DictReader((out / "null_control_contrasts.csv").open(encoding="utf-8")))
            rec = next(r for r in records if r["mask"] == "E3s0" and r["other"] == "E3s1")
            self.assertAlmostEqual(float(rec["wikitext2_ppl_delta_real"]), 0.10, places=6)
            self.assertAlmostEqual(float(rec["wikitext2_ppl_delta_null"]), 0.04, places=6)
            self.assertAlmostEqual(float(rec["wikitext2_ppl_excess"]), 0.06, places=6)
            self.assertEqual(rec["real_genes"], "3.0")
            self.assertEqual(rec["null_genes"], "5.0")
            summary = (out / "null_control_summary.md").read_text(encoding="utf-8")
            self.assertIn("E3s0", summary)


if __name__ == "__main__":
    unittest.main()
