import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_tiny_broader_database.sh"


class BroaderDatabaseLauncherTest(unittest.TestCase):
    def run_dry(self, method: str, scope: str) -> str:
        result = subprocess.run(
            ["bash", str(SCRIPT), method, scope, "--dry-run"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def test_attention_quant_scope(self):
        output = self.run_dry("quant", "attention")

        self.assertIn("quant_db_tinyllama_attention_bits234", output)
        self.assertIn("q\\|k\\|v\\|o", output)
        self.assertIn("--drop_saved_file_cache", output)

    def test_all_linear_sparse_scope(self):
        output = self.run_dry("sparse", "all-linear")

        self.assertIn("sparse_db_tinyllama_alllinear_s0.50", output)
        self.assertIn("gate\\|up\\|down", output)
        self.assertIn("--drop_saved_file_cache", output)


if __name__ == "__main__":
    unittest.main()
