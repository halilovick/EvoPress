import unittest

from scripts.summarize_adaptive_mutation_screen import paired_rows


class AdaptiveMutationSummaryTest(unittest.TestCase):
    def test_paired_rows_compute_adaptive_minus_baseline(self) -> None:
        rows = []
        for seed in (0, 1, 2):
            rows.extend(
                [
                    {
                        "variant": "baseline",
                        "seed": seed,
                        "wikitext2_ppl": 11.0 + seed,
                        "final_calibration_kl": 0.3 + seed,
                        "runtime_seconds": 100 + seed,
                    },
                    {
                        "variant": "adaptive",
                        "seed": seed,
                        "wikitext2_ppl": 10.5 + seed,
                        "final_calibration_kl": 0.2 + seed,
                        "runtime_seconds": 98 + seed,
                        "elevated_strength_generations": seed,
                        "elevated_strength_replacements": 0,
                    },
                    {
                        "variant": "fixed",
                        "seed": seed,
                        "wikitext2_ppl": 10.4 + seed,
                        "final_calibration_kl": 0.19 + seed,
                        "runtime_seconds": 99 + seed,
                    },
                ]
            )

        paired = paired_rows(rows)

        self.assertEqual(len(paired), 3)
        self.assertEqual(
            paired[0]["adaptive_minus_baseline_ppl"], -0.5
        )
        self.assertAlmostEqual(
            paired[0]["fixed_minus_baseline_ppl"], -0.6
        )
        self.assertAlmostEqual(
            paired[0]["adaptive_minus_fixed_ppl"], 0.1
        )
        self.assertAlmostEqual(
            paired[0]["adaptive_minus_baseline_kl"], -0.1
        )
        self.assertAlmostEqual(
            paired[0]["fixed_minus_baseline_kl"], -0.11
        )
        self.assertEqual(
            paired[0]["adaptive_minus_baseline_runtime_seconds"], -2
        )
        self.assertEqual(paired[2]["elevated_strength_generations"], 2)


if __name__ == "__main__":
    unittest.main()
