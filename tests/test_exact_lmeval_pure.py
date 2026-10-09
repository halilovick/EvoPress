"""Torch-free tests for the full-space LM-eval tooling (plan, helpers, summary)."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_exact_lmeval  # noqa: E402
import summarize_exact_lmeval  # noqa: E402
from src.exact_lmeval import (  # noqa: E402
    FULL_TASK_SIZES,
    check_settings,
    csv_rows,
    incomplete_tasks,
    require_smoke_dir_for_limit,
    settings_of,
    task_scores,
)

SCREENING_DENSE = REPO_ROOT / "results/runs/lmeval_dense_mistral_tasks_seed0_retry4/lmeval_results.json"
SMOKE_DENSE = REPO_ROOT / "results/runs/lmeval_dense_mistral_tasks_seed0_retry3/lmeval_results.json"
TASKS = ("arc_easy", "piqa", "winogrande")


def fake_args(**overrides):
    values = dict(base_model="mistralai/Mistral-7B-v0.3", tasks="arc_easy,piqa,winogrande", num_fewshot=0,
                  batch_size=4, limit=None, dtype="float16", attn_implementation="sdpa",
                  quant_db="/db", target_bitwidth=3)
    values.update(overrides)
    return argparse.Namespace(**values)


class PlanTests(unittest.TestCase):
    def test_default_plan_has_eleven_budget_feasible_jobs(self):
        plan = plan_exact_lmeval.build_plan(REPO_ROOT, list(plan_exact_lmeval.DEFAULT_SOURCES))
        ids = [job["id"] for job in plan["jobs"]]
        self.assertEqual(len(ids), 11)
        self.assertEqual(ids[:2], ["dense", "own_uniform3"])
        self.assertEqual(set(ids), {j for _, _, jobs in summarize_exact_lmeval.METHODS for j in jobs})
        for job in plan["jobs"]:
            if job["precision"] == "quantized":
                self.assertEqual(job["static_deficits"], [0, 0, 0])
                self.assertEqual(job["repair"]["scope"], "none")
        removed = {job["id"]: job.get("removed") for job in plan["jobs"]}
        self.assertEqual(removed["own_E2s1"], [0, 0])
        self.assertEqual(removed["own_J12s2"], [4, 4])
        self.assertEqual(removed["own_E3s0"], [8, 8])

    def test_plan_jobs_match_the_replay_tier0_definitions(self):
        replay_plan = json.loads((REPO_ROOT / "thesis_results/exact_replay_plan/replay_plan.json").read_text())
        replay_jobs = {job["id"]: job for job in replay_plan["jobs"] if job["tier"] == "0"}
        plan = plan_exact_lmeval.build_plan(REPO_ROOT, list(plan_exact_lmeval.DEFAULT_SOURCES))
        for job in plan["jobs"]:
            if job["id"] in replay_jobs:
                for key in ("mask", "bits", "repair", "precision"):
                    self.assertEqual(job[key], replay_jobs[job["id"]][key], (job["id"], key))
                label = job["mask"].get("source")
                if label:
                    self.assertEqual(plan["sources"][label], replay_plan["sources"][label])


class HelperTests(unittest.TestCase):
    def test_scores_from_a_real_screening_result(self):
        raw = json.loads(SCREENING_DENSE.read_text())
        scores = task_scores(raw, TASKS)
        self.assertEqual(scores["arc_easy"]["metric"], "acc_norm")
        self.assertEqual(scores["piqa"]["metric"], "acc_norm")
        self.assertEqual(scores["winogrande"]["metric"], "acc")
        self.assertAlmostEqual(scores["arc_easy"]["score"], raw["results"]["arc_easy"]["acc_norm,none"])
        self.assertAlmostEqual(scores["arc_easy"]["stderr"], raw["results"]["arc_easy"]["acc_norm_stderr,none"])
        self.assertIsNone(scores["winogrande"]["acc_norm"])
        self.assertEqual(incomplete_tasks(scores), {})
        self.assertEqual({t: s["n_samples"] for t, s in scores.items()}, FULL_TASK_SIZES)

    def test_limited_result_is_flagged_incomplete(self):
        raw = json.loads(SMOKE_DENSE.read_text())
        self.assertEqual(set(incomplete_tasks(task_scores(raw, TASKS))), set(TASKS))

    def test_limit_requires_smoke_directory(self):
        with self.assertRaises(SystemExit):
            require_smoke_dir_for_limit(Path("results/exact_lmeval/lmeval_20261009"), 0.02)
        require_smoke_dir_for_limit(Path("results/exact_lmeval/smoke_20261009"), 0.02)
        require_smoke_dir_for_limit(Path("results/exact_lmeval/lmeval_20261009"), None)

    def test_settings_cannot_change_within_one_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            check_settings(out, settings_of(fake_args(), "0.4.13"), overwrite=False)
            check_settings(out, settings_of(fake_args(quant_db="/other"), "0.4.13"), overwrite=False)
            with self.assertRaises(SystemExit):
                check_settings(out, settings_of(fake_args(batch_size=8), "0.4.13"), overwrite=False)
            with self.assertRaises(SystemExit):
                check_settings(out, settings_of(fake_args(), "0.4.14"), overwrite=False)


class SummaryTests(unittest.TestCase):
    def _write_results(self, out: Path, raw, *, cost=26982023168, limit=None, skip=()):
        plan = plan_exact_lmeval.build_plan(REPO_ROOT, list(plan_exact_lmeval.DEFAULT_SOURCES))
        settings = settings_of(fake_args(limit=limit), "0.4.13")
        for job in plan["jobs"]:
            if job["id"] in skip:
                continue
            removed = job.get("removed", [0, 0])
            details = {"removed_attn": list(range(removed[0])), "removed_mlp": list(range(removed[1]))}
            if job["precision"] == "quantized":
                details.update({"cost_bits": cost, "bits_sha256": "x"})
            result = {"job": job, "status": "completed", "settings": settings, "details": details,
                      "scores": task_scores(raw, TASKS), "seconds": 1.0}
            path = out / "jobs" / job["id"] / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
        return plan

    def test_complete_run_passes_all_checks_except_hashes(self):
        raw = json.loads(SCREENING_DENSE.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plan = self._write_results(out, raw)
            results = summarize_exact_lmeval.load_results(out)
            checks = {name: ok for name, ok, _ in summarize_exact_lmeval.verify(results, {}, raw)}
            self.assertTrue(all(checks.values()), checks)
            # A recorded replay hash that differs must fail the check.
            bad = summarize_exact_lmeval.verify(results, {"own_E3s0": "y"}, raw)
            self.assertFalse(next(ok for name, ok, _ in bad if name.startswith("bit-widths")))
            per_model, per_method = summarize_exact_lmeval.summarize(results)
            self.assertEqual(len(per_model), 11)
            self.assertEqual([m["method"] for m in per_method], ["E0", "E1", "E2", "J12", "E3"])
            self.assertEqual(len(csv_rows(out, plan)), 33)

    def test_missing_job_wrong_cost_and_limit_fail(self):
        raw = json.loads(SCREENING_DENSE.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write_results(out, raw, cost=1, limit=0.02, skip=("own_E3s2",))
            checks = {name.split(" (")[0]: ok for name, ok, _ in
                      summarize_exact_lmeval.verify(summarize_exact_lmeval.load_results(out), {}, raw)}
            self.assertFalse(checks["all 11 jobs completed"])
            self.assertFalse(checks["no sample limit"])
            self.assertFalse(checks["realized cost = 26,982,023,168 bits"])

    def test_real_replay_hashes_are_available_for_overlapping_jobs(self):
        hashes = summarize_exact_lmeval.replay_hashes(REPO_ROOT / "results/exact_replays/replay_20261008")
        for job in ("own_uniform3", "own_E3s0", "own_E3s1", "own_E3s2", "own_J12s0", "own_J12s1", "own_J12s2"):
            self.assertIn(job, hashes)


if __name__ == "__main__":
    unittest.main()


class IndependentSetTests(unittest.TestCase):
    """The six independent compositions (replay candidates)."""

    def setUp(self):
        self.plan = plan_exact_lmeval.build_independent_plan(REPO_ROOT)

    def test_jobs_are_copies_of_the_replay_jobs(self):
        replay_125 = json.loads((REPO_ROOT / "thesis_results/exact_replay_plan_125/replay_plan.json").read_text())
        replay_97 = json.loads((REPO_ROOT / "thesis_results/exact_replay_plan/replay_plan.json").read_text())
        jobs_125 = {j["id"]: j for j in replay_125["jobs"]}
        jobs_97 = {j["id"]: j for j in replay_97["jobs"]}
        self.assertEqual([j["id"] for j in self.plan["jobs"]], plan_exact_lmeval.INDEPENDENT_JOBS)
        for job in self.plan["jobs"]:
            for key in ("mask", "bits", "repair", "precision"):
                self.assertEqual(job[key], jobs_125[job["id"]][key])
                if job["id"] in jobs_97:  # the 25% compositions of the first batch
                    self.assertEqual(job[key], jobs_97[job["id"]][key])
            self.assertEqual(job["removed"], [4, 4] if "DO12" in job["id"] else [8, 8])
        for label, path in self.plan["sources"].items():
            self.assertEqual(path, replay_125["sources"][label])

    def test_replay_hashes_exist_for_all_six(self):
        hashes = {}
        for d in summarize_exact_lmeval.DEFAULT_REPLAY_DIRS:
            hashes.update(summarize_exact_lmeval.replay_hashes(REPO_ROOT / d, "ind_"))
        for job in plan_exact_lmeval.INDEPENDENT_JOBS:
            self.assertIn(job, hashes)

    def _write(self, out, status, digest_of, settings, raw=None):
        for job in self.plan["jobs"]:
            removed = job["removed"]
            result = {"job": job, "status": status, "settings": settings,
                      "details": {"removed_attn": list(range(removed[0])), "removed_mlp": list(range(removed[1])),
                                  "cost_bits": 26982023168, "bits_sha256": digest_of(job["id"])}}
            if raw is not None:
                result["scores"] = task_scores(raw, TASKS)
            path = out / "jobs" / job["id"] / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))

    def test_dry_run_check_requires_replay_hashes(self):
        hashes = {}
        for d in summarize_exact_lmeval.DEFAULT_REPLAY_DIRS:
            hashes.update(summarize_exact_lmeval.replay_hashes(REPO_ROOT / d, "ind_"))
        spec = summarize_exact_lmeval.SETS["independent"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, "validated", hashes.get, settings_of(fake_args(), "0.4.13"))
            checks = summarize_exact_lmeval.verify_candidates(
                summarize_exact_lmeval.load_results(out), hashes, spec["methods"], spec["removed"])
            self.assertTrue(all(ok for _, ok, _ in checks), checks)
            self._write(out, "validated", lambda j: "other", settings_of(fake_args(), "0.4.13"))
            checks = summarize_exact_lmeval.verify_candidates(
                summarize_exact_lmeval.load_results(out), hashes, spec["methods"], spec["removed"])
            self.assertFalse(all(ok for _, ok, _ in checks))

    def test_full_check_compares_settings_with_the_11_model_run(self):
        raw = json.loads(SCREENING_DENSE.read_text())
        hashes = {}
        for d in summarize_exact_lmeval.DEFAULT_REPLAY_DIRS:
            hashes.update(summarize_exact_lmeval.replay_hashes(REPO_ROOT / d, "ind_"))
        spec = summarize_exact_lmeval.SETS["independent"]
        reference = json.loads((REPO_ROOT / summarize_exact_lmeval.DEFAULT_REFERENCE_SETTINGS).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            same = dict(reference, quant_db="/elsewhere")
            self._write(out, "completed", hashes.get, same, raw)
            checks = summarize_exact_lmeval.verify(
                summarize_exact_lmeval.load_results(out), hashes, None, methods=spec["methods"],
                want=spec["removed"], reference_settings=reference, require_replay=True)
            self.assertTrue(all(ok for _, ok, _ in checks), checks)
            self._write(out, "completed", hashes.get, dict(reference, batch_size=8), raw)
            checks = dict((n, ok) for n, ok, _ in summarize_exact_lmeval.verify(
                summarize_exact_lmeval.load_results(out), hashes, None, methods=spec["methods"],
                want=spec["removed"], reference_settings=reference, require_replay=True))
            self.assertFalse(checks["settings identical to the 11-model LM-eval"])
