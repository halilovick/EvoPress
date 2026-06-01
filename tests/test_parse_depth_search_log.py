import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PARSER = REPO_ROOT / "scripts" / "parse_depth_search_log.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "depth_search_sample.txt"


class ParseDepthSearchLogTest(unittest.TestCase):
    def run_parser(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(PARSER), *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def test_parses_all_generations_and_final_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "fixture_depth_run" / "generation_metrics.csv"
            result = self.run_parser(
                "--log",
                str(FIXTURE),
                "--output",
                str(output),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Parsed 5/5 generations for fixture_depth_run.", result.stdout)
            self.assertIn("Final WikiText2 PPL: 85.00", result.stdout)

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["phase"], "generation")
            self.assertEqual(rows[0]["generation"], "1")
            self.assertEqual(rows[0]["train_fitness"], "7.50e-01")
            self.assertEqual(rows[0]["parent_attn_mask"], "[1,0,0,1]")
            self.assertEqual(rows[4]["wikitext2_ppl"], "92.00")
            self.assertEqual(rows[5]["phase"], "final")
            self.assertEqual(rows[5]["generation"], "")
            self.assertEqual(rows[5]["wikitext2_ppl"], "85.00")
            self.assertEqual(rows[5]["train_ppl"], "8.40e+01")
            self.assertEqual(rows[5]["parent_attn_mask"], "[0,0,1,1]")
            self.assertEqual(rows[5]["parent_mlp_mask"], "[0,1,0,1]")

    def test_rejects_incomplete_log_unless_explicitly_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            incomplete_log = temp_path / "incomplete.txt"
            incomplete_log.write_text(
                "\n".join(
                    [
                        "Generation 1/2",
                        "Train fitness 7.50e-01",
                        "Parent: attn: [1, 0] mlp: [0, 1]",
                        "wikitext2: 994.00",
                        "full train ppl: 9.80e+02",
                    ]
                ),
                encoding="utf-8",
            )
            output = temp_path / "partial_run" / "generation_metrics.csv"

            rejected = self.run_parser("--log", str(incomplete_log), "--output", str(output))
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Incomplete generation coverage", rejected.stderr)
            self.assertFalse(output.exists())

            allowed = self.run_parser(
                "--log",
                str(incomplete_log),
                "--output",
                str(output),
                "--allow-incomplete",
            )
            self.assertEqual(allowed.returncode, 0, allowed.stderr)
            self.assertIn("Parsed 1/2 generations for partial_run.", allowed.stdout)
            self.assertIn("Incomplete generation coverage", allowed.stderr)

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["generation"], "1")


if __name__ == "__main__":
    unittest.main()
