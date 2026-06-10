import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from src.run_reporting import (
    RunReporter,
    build_depth_details,
    build_final_candidate,
    compute_compression_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "validate_run_outputs.py"


class Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.o_proj = torch.nn.Linear(4, 4, bias=False)


class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(4, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 4, bias=False)


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attention()
        self.mlp = MLP()


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(10, 4)
        self.layers = torch.nn.ModuleList([Block(), Block()])
        self.lm_head = torch.nn.Linear(4, 10, bias=False)


class RunReportingTest(unittest.TestCase):
    def test_compression_accounting_and_validator(self) -> None:
        model = FakeModel().half()
        attention_names = [
            "layers.0.self_attn",
            "layers.1.self_attn",
        ]
        mlp_names = [
            "layers.0.mlp",
            "layers.1.mlp",
        ]
        drop_state = {
            "attn": [True, False],
            "mlp": [False, False],
        }
        bitwidths = {
            "layers.0.self_attn.q_proj": 2,
            "layers.1.self_attn.q_proj": 4,
        }

        depth = build_depth_details(attention_names, mlp_names, drop_state)
        compression = compute_compression_metrics(model, depth, bitwidths)

        parameters = compression["parameter_statistics"]
        quant = compression["quantization_statistics"]
        size = compression["model_size_statistics"]

        self.assertEqual(parameters["total_parameters_dense"], 272)
        self.assertEqual(parameters["dropped_parameters"], 32)
        self.assertEqual(parameters["active_parameters"], 240)
        self.assertEqual(parameters["searched_parameters_dense"], 32)
        self.assertEqual(parameters["searched_parameters_active"], 16)
        self.assertEqual(quant["quantized_module_count"], 2)
        self.assertEqual(quant["active_quantized_module_count"], 1)
        self.assertEqual(quant["bitwidth_histogram"], {"4": 1})
        self.assertEqual(quant["average_bitwidth_active"], 4.0)
        self.assertAlmostEqual(size["estimated_weight_memory_mb"] * 1024**2 * 8, 3648.0)
        self.assertGreater(size["estimated_compression_ratio"], 1.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "reporting_fixture"
            reporter = RunReporter(run_dir, "joint_depth_quant", repo_root=REPO_ROOT)
            reporter.append_generation(
                {
                    "generation": 1,
                    "best_search_fitness": 0.5,
                    "fitness_fn": "kl",
                    "best_calibration_kl": None,
                    "best_train_ppl": 10.0,
                    "wikitext2_ppl": 11.0,
                    "c4_ppl": None,
                    "fineweb_edu_ppl": None,
                    "eval_tokens_used": 1024,
                    "eval_tokens_by_dataset": {"wikitext2": 1024},
                    "num_offspring": 2,
                    "num_survivors_stage_1": 1,
                    "num_survivors_stage_2": None,
                    "num_survivors_stage_3": None,
                    "survivors_per_selection": [1],
                    "tokens_per_selection": [512],
                    "active_parameters": parameters["active_parameters"],
                    "average_bitwidth_active": quant["average_bitwidth_active"],
                    "estimated_weight_memory_mb": size["estimated_weight_memory_mb"],
                    "dropped_attention_count": 1,
                    "dropped_mlp_count": 0,
                    "mutation_summary": {"depth": 1, "quantization": 1},
                    "accepted_parent_replacement": True,
                    "runtime_seconds_cumulative": 1.0,
                    "peak_gpu_memory_mb": None,
                }
            )
            candidate = build_final_candidate(
                "joint_depth_quant",
                depth,
                bitwidths,
                {"drop": drop_state, "quant": [[2, 4]]},
            )
            candidate_path = reporter.write_final_candidate(candidate)
            reporter.write_summary(
                model_name="fixture/model",
                dataset_calibration="fixture",
                dataset_eval=["wikitext2"],
                search_config={
                    "generations": 1,
                    "offspring": 2,
                    "initial_candidates": 2,
                    "selection_tokens": [512],
                    "selection_survivors": [1],
                    "fitness_fn": "kl",
                    "sequence_length": 512,
                    "calibration_tokens": 1024,
                    "seed": 0,
                },
                compression_config={
                    "target_depth_sparsity": 0.5,
                    "target_average_bitwidth": 3.0,
                    "bits_available": [2, 3, 4],
                    "group_size": 128,
                },
                final_metrics={
                    "best_search_fitness": 0.5,
                    "final_calibration_kl": 0.4,
                    "wikitext2_ppl": 11.0,
                    "train_ppl": 10.0,
                },
                parameter_statistics=parameters,
                depth_statistics={
                    key: value
                    for key, value in depth.items()
                    if key not in {"kept_modules", "attention_mask", "mlp_mask"}
                },
                quantization_statistics=quant,
                model_size_statistics=size,
                artifacts={
                    "candidate_path": candidate_path,
                    "generation_log_path": str(run_dir / "generation_log.csv"),
                    "config_path": str(run_dir / "joint_config.json"),
                    "stdout_log_path": str(run_dir / "run.log"),
                },
            )

            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["final_metrics"]["active_parameters"], 240)
            self.assertEqual(summary["depth_statistics"]["dropped_attention_count"], 1)

            result = subprocess.run(
                [sys.executable, "-B", str(VALIDATOR), str(run_dir)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("OK: run_summary.json", result.stdout)


if __name__ == "__main__":
    unittest.main()
