import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_mistral_medium_composition_grid.sh"


class RunMistralMediumCompositionGridTest(unittest.TestCase):
    def test_attention_scope_dry_run_uses_custom_source_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            quant_db = temp_path / "quant_db" / "Mistral-7B-v0.3" / "3bit"
            source_runs = temp_path / "runs"

            for module_index in range(128):
                module_dir = quant_db / f"module_{module_index}"
                module_dir.mkdir(parents=True)
                for bitwidth in (2, 3, 4):
                    (module_dir / f"{bitwidth}.pth").write_text(
                        "fixture\n", encoding="utf-8"
                    )

            depth_dir = (
                source_runs
                / "thesis_medium_depth_mistral_s0.25_g20_o16_seed0"
            )
            quant_dir = (
                source_runs
                / "thesis_attention_quant_mistral_attention3.0_g20_o16_seed0"
            )
            depth_dir.mkdir(parents=True)
            quant_dir.mkdir(parents=True)
            (depth_dir / "layer_drop_config.txt").write_text(
                "none\n", encoding="utf-8"
            )
            (quant_dir / "quant_configuration.txt").write_text(
                "3\n", encoding="utf-8"
            )

            result = subprocess.run(
                [str(LAUNCHER), "--dry-run"],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "QUANT_WEIGHTS_PATH": str(quant_db),
                    "SOURCE_RUNS_ROOT": str(source_runs),
                    "RESULTS_RUNS_ROOT": str(temp_path / "synced_runs"),
                    "OUTPUTS_ROOT": str(temp_path / "outputs"),
                    "EXPERIMENT_LOG": str(temp_path / "experiment_log.csv"),
                    "DEPTH_SOURCE_PREFIX": "thesis_medium",
                    "QUANT_SOURCE_PREFIX": "thesis_attention",
                    "RUN_PREFIX": "thesis_attention",
                    "QUANT_SCOPE_LABEL": "attention",
                    "EXPECTED_QUANT_MODULES": "128",
                    "EXPECTED_QUANT_WEIGHT_FILES": "384",
                    "SEEDS": "0",
                    "MODES": "independent uniform",
                },
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "thesis_attention_independent_depth_quant_mistral_s0.25_attention3.0_seed0",
                result.stdout,
            )
            self.assertIn(
                "thesis_attention_depth_uniform_quant_mistral_s0.25_attention3.0_seed0",
                result.stdout,
            )
            self.assertIn(str(depth_dir / "layer_drop_config.txt"), result.stdout)
            self.assertIn(str(quant_dir / "quant_configuration.txt"), result.stdout)
            self.assertIn(f"--quant_weights_path {quant_db}", result.stdout)


if __name__ == "__main__":
    unittest.main()
