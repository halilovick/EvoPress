import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
LMEVAL_PATH = REPO_ROOT / "lmeval.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class LmEvalTaskCompatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.saved_modules = {
            name: sys.modules.get(name)
            for name in [
                "lm_eval",
                "lm_eval.evaluator",
                "lm_eval.utils",
                "lm_eval.tasks",
                "lm_eval.api",
                "lm_eval.api.registry",
            ]
        }

        lm_eval = types.ModuleType("lm_eval")
        evaluator = types.ModuleType("lm_eval.evaluator")
        utils = types.ModuleType("lm_eval.utils")
        tasks = types.ModuleType("lm_eval.tasks")
        api = types.ModuleType("lm_eval.api")
        registry = types.ModuleType("lm_eval.api.registry")

        def pattern_match(patterns, source):
            return [item for item in source if item in patterns]

        def simple_evaluate(**kwargs):
            return {"kwargs": kwargs, "config": {"batch_sizes": [1]}}

        class TaskManager:
            def __init__(self, *args, **kwargs):
                self.all_tasks = ["arc_easy", "piqa", "winogrande"]

        evaluator.simple_evaluate = simple_evaluate
        utils.eval_logger = logging.getLogger("lm_eval_fixture")
        utils.pattern_match = pattern_match
        utils.load_yaml_config = lambda path: {"yaml_path": path}
        utils.SPACING = "  "
        utils.make_table = lambda results, group=None: ""
        tasks.TaskManager = TaskManager
        tasks.include_path = lambda path: None
        tasks.initialize_tasks = lambda verbosity: None
        lm_eval.evaluator = evaluator
        lm_eval.utils = utils

        sys.modules["lm_eval"] = lm_eval
        sys.modules["lm_eval.evaluator"] = evaluator
        sys.modules["lm_eval.utils"] = utils
        sys.modules["lm_eval.tasks"] = tasks
        sys.modules["lm_eval.api"] = api
        sys.modules["lm_eval.api.registry"] = registry

    def tearDown(self) -> None:
        for name, module in self.saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.modules.pop("lmeval_task_compat_fixture", None)

    def load_module(self):
        spec = importlib.util.spec_from_file_location(
            "lmeval_task_compat_fixture",
            LMEVAL_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_import_does_not_require_all_tasks_export(self) -> None:
        module = self.load_module()

        self.assertIsNone(module.ALL_TASKS)
        task_manager = module.make_task_manager(
            SimpleNamespace(verbosity="INFO", include_path=None)
        )
        tasks = module.resolve_tasks(
            SimpleNamespace(
                tasks="arc_easy,piqa",
                verbosity="INFO",
                include_path=None,
            ),
            module.utils.eval_logger,
            task_manager,
        )

        self.assertEqual(tasks, ["arc_easy", "piqa"])

    def test_simple_evaluate_compat_passes_task_manager_when_supported(self) -> None:
        module = self.load_module()

        def simple_evaluate(*, model, tasks, task_manager=None):
            return {
                "model": model,
                "tasks": tasks,
                "task_manager": task_manager,
            }

        module.evaluator.simple_evaluate = simple_evaluate
        manager = object()
        result = module.simple_evaluate_compat(
            manager,
            model="hf",
            tasks=["arc_easy"],
        )

        self.assertIs(result["task_manager"], manager)

    def test_get_eval_logger_falls_back_when_utils_logger_is_missing(self) -> None:
        module = self.load_module()
        delattr(module.utils, "eval_logger")

        logger = module.get_eval_logger("INFO")

        self.assertIs(logger, logging.getLogger("lm_eval"))
        self.assertEqual(logger.level, logging.INFO)


if __name__ == "__main__":
    unittest.main()
