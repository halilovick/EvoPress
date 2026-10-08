"""CPU tests of the replay runner's candidate construction (meta-tensor model).

Requires PyTorch and transformers (the runner's imports), but no model weights
or GPU. The quantization database is simulated by empty level files.
"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evo_exact_replay as runner  # noqa: E402
from scripts import plan_exact_replays as planner  # noqa: E402
from src import exact_replay as er  # noqa: E402
from src.compression_budget import uniform_quantization_target_cost  # noqa: E402
from src.model_utils import group_layers, layer_order_fn  # noqa: E402
from test_compression_budget import Mistral7BV03Shape  # noqa: E402


class ExactReplayRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        root = Path(cls.tmp.name)
        cls.db = root / "db"
        cls.model = Mistral7BV03Shape()
        names = []
        for name, module in cls.model.named_modules():
            if name.endswith("_proj"):
                names.append(name)
                (cls.db / name).mkdir(parents=True)
                for level in range(2, 7):
                    (cls.db / name / f"{level}.pth").touch()
        names.sort(key=layer_order_fn)
        cls.grouped = group_layers(cls.model, names, "size")
        planner.main(["--output_dir", str(root / "plan")])
        cls.plan = json.loads((root / "plan" / "replay_plan.json").read_text())
        cls.args = runner.parse_args([
            "--plan", str(root / "plan" / "replay_plan.json"),
            "--quant_db", str(cls.db), "--output_dir", str(root / "out"),
        ])
        cost_kwargs = {
            "attention_module_names": [f"model.layers.{i}.self_attn" for i in range(32)],
            "mlp_module_names": [f"model.layers.{i}.mlp" for i in range(32)],
            "dense_dtype_bits": 16, "group_size": 128, "include_quantization_metadata": True,
            "scale_bits": 16, "zero_point_bits": 16,
        }
        target = int(uniform_quantization_target_cost(cls.model, cls.grouped, 3, **cost_kwargs)["total_cost_bits"])
        assert target == 26_982_023_168
        cls.ctx = {
            "model": cls.model, "grouped_layer_names": cls.grouped,
            "sources": {k: er.load_final_candidate(REPO_ROOT / v) for k, v in cls.plan["sources"].items()},
            "num_layers": 32, "cost_kwargs": cost_kwargs, "target_cost_bits": target,
            "bi_scores": {"attn": [float(i % 7) for i in range(32)], "mlp": [float((5 * i) % 11) for i in range(32)]},
        }

    def jobs(self, predicate):
        return [job for job in self.plan["jobs"] if predicate(job)]

    def test_every_quantized_job_builds_at_the_exact_budget(self):
        for job in self.jobs(lambda j: j["precision"] == "quantized"):
            candidate, details = runner.build_candidate(job, self.args, self.ctx)
            self.assertEqual(details["cost_bits"], self.ctx["target_cost_bits"], job["id"])
            self.assertTrue(details["repair_changed_active_only"], job["id"])
            if job.get("repair", {}).get("scope") == "none":
                self.assertEqual(details["repair_changed_genes"], 0, job["id"])

    def test_exclusive_repair_touches_only_exclusive_sublayers(self):
        for job in self.jobs(lambda j: (j.get("repair") or {}).get("scope") == "exclusive"):
            _, details = runner.build_candidate(job, self.args, self.ctx)
            mask = self.ctx["sources"][job["mask"]["source"]]["drop"]
            donor = self.ctx["sources"][job["bits"]["source"]]["drop"]
            exclusive = er.exclusive_sublayers(mask, donor)
            for name in details["repair_changes"]:
                kind = er.module_kind(name)
                self.assertTrue(exclusive[kind][er.layer_index(name)], (job["id"], name))
            if any(job["static_deficits"]):
                self.assertGreater(details["repair_changed_genes"], 0)

    def test_repair_free_shared_gene_exchange_changes_nothing(self):
        for job in self.jobs(lambda j: j["tier"] == "A" and j["pair"]["fill"] == "owner"):
            _, details = runner.build_candidate(job, self.args, self.ctx)
            self.assertEqual(details["repair_changed_genes"], 0)
            mask_src = self.ctx["sources"][job["mask"]["source"]]
            donor = self.ctx["sources"][job["bits"]["source"]]
            for name, level in details["bits"].items():
                if er.module_is_active(name, mask_src["drop"]) and er.module_is_active(name, donor["drop"]):
                    self.assertEqual(level, donor["bits"][name])

    def test_repair_is_deterministic_per_seed(self):
        job = next(j for j in self.plan["jobs"] if j["id"].startswith("x_donor_all_J12s0_QJ12s1_r0"))
        first = runner.build_candidate(job, self.args, self.ctx)[1]["bits_sha256"]
        again = runner.build_candidate(copy.deepcopy(job), self.args, self.ctx)[1]["bits_sha256"]
        self.assertEqual(first, again)

    def test_own_jobs_reproduce_final_candidates(self):
        for job in self.jobs(lambda j: j["id"].startswith("own_") and "source" in j["mask"]):
            _, details = runner.build_candidate(job, self.args, self.ctx)
            self.assertEqual(details["bits"], self.ctx["sources"][job["mask"]["source"]]["bits"])

    def test_fp16_jobs_carry_no_bit_widths_and_run_first(self):
        for job in self.jobs(lambda j: j["precision"] == "fp16"):
            candidate, details = runner.build_candidate(job, self.args, self.ctx)
            self.assertIsNone(candidate["quant"])
            self.assertEqual(len(details["removed_attn"]), len(details["removed_mlp"]))
        ordered = runner.select_jobs(self.plan, self.args)
        kinds = [job["precision"] for job in ordered]
        self.assertEqual(kinds, sorted(kinds, key=lambda k: k != "fp16"))

    def test_bi_score_mask_uses_lowest_scores(self):
        job = next(j for j in self.plan["jobs"] if j["mask"].get("rule") == "bi_score" and j["mask"]["k_attn"] == 8)
        drop = runner.resolve_mask(job, self.ctx["sources"], 32, self.ctx["bi_scores"])
        self.assertEqual(drop, er.score_mask(self.ctx["bi_scores"], 8, 8))

    def test_result_row_has_all_columns(self):
        job = self.plan["jobs"][0]
        row = runner.result_row(job, {"status": "completed", "metrics": {"wikitext2": 8.0}, "details": {}})
        self.assertEqual(set(row), set(runner.RESULT_COLUMNS))


if __name__ == "__main__":
    unittest.main()
