import unittest
from pathlib import Path

from scripts.summarize_depth_warmstart_g50 import (
    CONDITION_ORDER,
    aggregate_rows,
    normalize_depth_candidate,
    paired_delta_rows,
    selection_cost,
    stable_json_hash,
    validate_final_candidate,
    validate_warm_provenance,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "results" / "runs"


def synthetic_rows() -> list[dict[str, object]]:
    ppls = {
        "standard_standard": (11.4, 11.2, 11.3),
        "depthwarm_standard": (11.1, 11.0, 11.2),
        "standard_interaction": (11.0, 11.1, 11.2),
        "depthwarm_interaction": (10.9, 11.0, 11.0),
    }
    rows: list[dict[str, object]] = []
    for condition in CONDITION_ORDER:
        warm = condition.startswith("depthwarm")
        for seed, ppl in enumerate(ppls[condition]):
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "wikitext2_ppl": ppl,
                    "best_search_fitness": 0.7 + seed / 100,
                    "stage1_runtime_minutes": 9.0 if warm else 0.0,
                    "stage2_runtime_minutes": 20.0 + seed,
                    "total_runtime_minutes": 29.0 + seed if warm else 20.0 + seed,
                    "stage1_candidate_evaluations": 572 if warm else 0,
                    "stage2_candidate_evaluations": 1351 if warm else 1382,
                    "total_candidate_evaluations": 1923 if warm else 1382,
                    "stage1_evaluated_tokens": 999424 if warm else 0,
                    "stage2_evaluated_tokens": 2458112 if warm else 2473984,
                    "total_evaluated_tokens": 3457536 if warm else 2473984,
                    "initial_candidate_evaluations": 1 if warm else 32,
                    "final_wikitext2_eval_tokens_loaded": 333824,
                }
            )
    return rows


class SummarizeDepthWarmstartG50Test(unittest.TestCase):
    def test_selection_cost_counts_all_stages_and_final_elitism(self) -> None:
        search = {
            "generations": 50,
            "offspring": 16,
            "initial_candidates": 32,
            "initial_tokens": 512,
            "selection_survivors": [8, 2, 1],
            "selection_tokens": [512, 2048, 8192],
        }

        cost = selection_cost(search)

        self.assertEqual(cost["candidate_evaluations_per_generation"], 27)
        self.assertEqual(cost["evaluated_tokens_per_generation"], 49152)
        self.assertEqual(cost["candidate_evaluations"], 1382)
        self.assertEqual(cost["evaluated_tokens"], 2473984)

    def test_warm_selection_cost_uses_actual_single_initial_candidate(self) -> None:
        search = {
            "generations": 50,
            "offspring": 16,
            "initial_candidates": 32,
            "initial_candidates_evaluated": 1,
            "initial_tokens": 512,
            "selection_survivors": [8, 2, 1],
            "selection_tokens": [512, 2048, 8192],
        }

        cost = selection_cost(search)

        self.assertEqual(cost["candidate_evaluations"], 1351)
        self.assertEqual(cost["evaluated_tokens"], 2458112)

    def test_aggregate_and_paired_rows_answer_the_four_comparisons(self) -> None:
        rows = synthetic_rows()

        aggregate = aggregate_rows(rows)
        paired = paired_delta_rows(rows)

        self.assertEqual(len(aggregate), 4)
        self.assertEqual(len(paired), 12)
        standard_warm = [
            row
            for row in paired
            if row["comparison"] == "warm_vs_standard_initialization__standard_mutation"
        ]
        self.assertEqual(
            [
                round(float(row["delta_ppl_method_minus_baseline"]), 3)
                for row in standard_warm
            ],
            [-0.3, -0.2, -0.1],
        )
        self.assertTrue(all(row["winner"] == "method" for row in standard_warm))

    def test_existing_stage_one_hash_and_warm_metadata_validate(self) -> None:
        stage1_dir = RUNS_ROOT / "thesis_medium_depth_mistral_s0.25_g20_o16_seed0"
        warm_dir = RUNS_ROOT / (
            "thesis_sequential_depth_to_joint_warm_mistral_s0.25_qproj3.0_g20_o16_seed0"
        )
        depth = normalize_depth_candidate(stage1_dir / "final_candidate.json")
        summary = __import__("json").loads(
            (warm_dir / "run_summary.json").read_text(encoding="utf-8")
        )

        provenance = validate_warm_provenance(
            summary,
            0,
            RUNS_ROOT,
            warm_dir,
        )

        self.assertEqual(
            stable_json_hash(depth),
            "454b86987800d97eba43ad3d810527ff7143b7cabdea85e154ab2d26a1831402",
        )
        self.assertEqual(
            provenance["initial_parent_hash"],
            "d317f58ebca1b047b5be28028314aceae4264823132a18e538debc4c1382bc9b",
        )

    def test_existing_standard_final_candidate_has_exact_active_budget(self) -> None:
        validate_final_candidate(
            RUNS_ROOT
            / ("thesis_compute_matched_joint_mistral_s0.25_qproj3.0_g50_o16_seed0")
            / "final_candidate.json"
        )


if __name__ == "__main__":
    unittest.main()
