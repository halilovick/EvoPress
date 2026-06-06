import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PARSER = REPO_ROOT / "scripts" / "parse_quant_search_log.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "quant_search_sample.txt"


class ParseQuantSearchLogTest(unittest.TestCase):
    def run_parser(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(PARSER), *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def test_parses_generations_bit_budget_and_final_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "quant_fixture" / "generation_metrics.csv"
            result = self.run_parser("--log", str(FIXTURE), "--output", str(output))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Parsed 2/2 generations for quant_fixture.", result.stdout)
            self.assertIn("Final WikiText2 PPL: 9.10", result.stdout)

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["phase"], "generation")
            self.assertEqual(rows[0]["search_point"], "[[3,3,3,3]]")
            self.assertEqual(rows[0]["bit_average"], "3.0000e+00")
            self.assertEqual(rows[0]["parent_bits"], "1200")
            self.assertEqual(rows[1]["selection_train_fitness"], "1.25e-02")
            self.assertEqual(rows[2]["phase"], "final")
            self.assertEqual(rows[2]["search_point"], "[[2,4,2,4]]")
            self.assertEqual(rows[2]["wikitext2_ppl"], "9.10")
            self.assertEqual(rows[2]["train_ppl"], "9.40")

    def test_rejects_incomplete_generation_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            log = temp_path / "incomplete.log"
            log.write_text(
                "\n".join(
                    [
                        "Generation 1/2",
                        "Current search point:",
                        "[3, 3]",
                        "Parent bits: 600",
                        "Bit average: 3.0000e+00",
                        "Train fitness: inf",
                    ]
                ),
                encoding="utf-8",
            )
            output = temp_path / "generation_metrics.csv"

            result = self.run_parser("--log", str(log), "--output", str(output))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Incomplete generation coverage", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
