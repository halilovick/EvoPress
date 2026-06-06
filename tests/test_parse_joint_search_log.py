import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PARSER = REPO_ROOT / "scripts" / "parse_joint_search_log.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "joint_search_sample.txt"


class ParseJointSearchLogTest(unittest.TestCase):
    def run_parser(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(PARSER), *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def test_parses_joint_generations_and_final_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "joint_fixture" / "generation_metrics.csv"
            result = self.run_parser("--log", str(FIXTURE), "--output", str(output))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Parsed 2/2 generations for joint_fixture.", result.stdout)
            self.assertIn("Final WikiText2 PPL: 11.70", result.stdout)

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["drop_config"], '["none","attn","mlp","none"]')
            self.assertEqual(rows[0]["quant_state"], "[[3,3,3,3]]")
            self.assertEqual(rows[0]["dropped_attn_modules"], "1")
            self.assertEqual(rows[0]["dropped_mlp_modules"], "1")
            self.assertEqual(rows[2]["phase"], "final")
            self.assertEqual(rows[2]["wikitext2_ppl"], "11.70")
            self.assertEqual(rows[2]["train_ppl"], "12.00")
            self.assertEqual(rows[2]["quant_bit_average"], "3.0000e+00")
            self.assertEqual(rows[2]["quant_state"], "[[2,4,2,4]]")


if __name__ == "__main__":
    unittest.main()
