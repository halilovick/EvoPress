import unittest

from scripts.summarize_joint_aware_probability_screen import (
    mutation_rows,
    paired_rows,
)


class JointAwareProbabilitySummaryTest(unittest.TestCase):
    def test_paired_rows_compute_direction_as_p025_minus_baseline(self) -> None:
        rows = []
        for seed in (0, 1, 2):
            rows.extend(
                [
                    {
                        "variant": "baseline",
                        "seed": seed,
                        "wikitext2_ppl": 10.0 + seed,
                        "final_calibration_kl": 0.2 + seed,
                        "runtime_seconds": 100 + seed,
                    },
                    {
                        "variant": "p025",
                        "seed": seed,
                        "wikitext2_ppl": 9.5 + seed,
                        "final_calibration_kl": 0.1 + seed,
                        "runtime_seconds": 105 + seed,
                    },
                ]
            )

        paired = paired_rows(rows)

        self.assertEqual(len(paired), 3)
        self.assertEqual(paired[0]["p025_minus_baseline_ppl"], -0.5)
        self.assertAlmostEqual(
            paired[0]["p025_minus_baseline_kl"], -0.1
        )
        self.assertEqual(
            paired[0]["p025_minus_baseline_runtime_seconds"], 5
        )

    def test_mutation_rows_sum_generated_and_selected_counts(self) -> None:
        rows = []
        for variant in ("baseline", "p025"):
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "variant": variant,
                        "generated_depth": seed + 1,
                        "generated_quantization": 2,
                        "generated_joint_aware": (
                            seed if variant == "p025" else 0
                        ),
                        "selected_depth": 1,
                        "selected_quantization": seed,
                        "selected_joint_aware": (
                            1 if variant == "p025" else 0
                        ),
                    }
                )

        mutations = mutation_rows(rows)
        aware = {
            row["mutation_type"]: row
            for row in mutations
            if row["variant"] == "p025"
        }

        self.assertEqual(aware["depth"]["generated_offspring"], 6)
        self.assertEqual(aware["quantization"]["selected_as_parent"], 3)
        self.assertEqual(aware["joint_aware"]["generated_offspring"], 3)
        self.assertEqual(aware["joint_aware"]["selected_as_parent"], 3)
        self.assertEqual(aware["joint_aware"]["selection_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
