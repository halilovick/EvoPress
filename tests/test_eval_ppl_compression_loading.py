import unittest
import importlib.machinery
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)
    datasets_stub.load_dataset = Mock()
    sys.modules["datasets"] = datasets_stub

import eval_ppl


class EvalPplCompressionLoadingTest(unittest.TestCase):
    def test_applies_sparse_quant_then_depth(self) -> None:
        args = SimpleNamespace(
            sparse_weights_path="sparse-db",
            sparse_config_path="sparse-config.txt",
            sparse_default_level=1,
            quant_weights_path="quant-db",
            quant_config_path="quant-config.txt",
            quant_default_level=2,
            drop_layer_config="drop-config.txt",
        )
        model = object()

        events: list[tuple[str, tuple[object, ...]]] = []

        def fake_load(*call_args: object) -> object:
            events.append(("load", call_args))
            return model

        def fake_drop(*call_args: object) -> object:
            events.append(("drop", call_args))
            return model

        with (
            patch.object(eval_ppl, "load_compressed_weights", side_effect=fake_load),
            patch.object(eval_ppl, "drop_layers_from_config", side_effect=fake_drop),
        ):
            returned = eval_ppl.apply_compression(args, model)

        self.assertIs(returned, model)
        self.assertEqual(
            events,
            [
                ("load", (model, "sparse-db", "sparse-config.txt", 1)),
                ("load", (model, "quant-db", "quant-config.txt", 2)),
                ("drop", (model, "drop-config.txt")),
            ],
        )

    def test_skips_missing_components_without_fallback_else_branch(self) -> None:
        args = SimpleNamespace(
            sparse_weights_path=None,
            sparse_config_path=None,
            sparse_default_level=0,
            quant_weights_path=None,
            quant_config_path=None,
            quant_default_level=0,
            drop_layer_config="drop-config.txt",
        )
        model = object()

        with (
            patch.object(eval_ppl, "load_compressed_weights", Mock()) as load_mock,
            patch.object(eval_ppl, "drop_layers_from_config", Mock(return_value=model)) as drop_mock,
        ):
            returned = eval_ppl.apply_compression(args, model)

        self.assertIs(returned, model)
        load_mock.assert_not_called()
        drop_mock.assert_called_once_with(model, "drop-config.txt")


if __name__ == "__main__":
    unittest.main()
