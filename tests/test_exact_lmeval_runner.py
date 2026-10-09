"""Environment checks for evo_exact_lmeval.py (needs torch, transformers, lm_eval).

Run on DataLab before the GPU job. No model is loaded.
"""

import argparse
import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HAVE_STACK = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "lm_eval"))


@unittest.skipUnless(HAVE_STACK, "needs torch, transformers and lm_eval")
class LmEvalApiTests(unittest.TestCase):
    def test_lm_eval_version_is_the_screening_version(self):
        from importlib.metadata import version

        self.assertEqual(version("lm_eval"), "0.4.13")

    def test_hflm_accepts_an_initialized_model(self):
        from lm_eval.models.huggingface import HFLM

        params = inspect.signature(HFLM.__init__).parameters
        for name in ("pretrained", "tokenizer", "batch_size"):
            self.assertIn(name, params)

    def test_simple_evaluate_arguments(self):
        from lm_eval import evaluator

        params = inspect.signature(evaluator.simple_evaluate).parameters
        for name in ("model", "tasks", "num_fewshot", "limit", "log_samples", "task_manager"):
            self.assertIn(name, params)

    def test_tasks_are_registered(self):
        from lmeval import all_task_names, make_task_manager

        manager = make_task_manager(argparse.Namespace(verbosity="ERROR", include_path=None))
        names = set(all_task_names(manager) or [])
        for task in ("arc_easy", "piqa", "winogrande"):
            self.assertIn(task, names)

    def test_runner_imports_and_parses(self):
        import evo_exact_lmeval

        args = evo_exact_lmeval.parse_args(["--plan", "p.json", "--quant_db", "/db", "--output_dir", "o"])
        self.assertEqual((args.num_fewshot, args.batch_size, args.limit, args.attn_implementation),
                         (0, 4, None, "sdpa"))
        self.assertEqual(args.tasks, "arc_easy,piqa,winogrande")
        self.assertFalse(args.allow_repair)
        args = evo_exact_lmeval.parse_args(["--plan", "p.json", "--quant_db", "/db", "--output_dir", "o",
                                            "--allow_repair"])
        self.assertTrue(args.allow_repair)


if __name__ == "__main__":
    unittest.main()
