import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from scripts.smoke_stage1_memory_path import (
    EXPECTED_ARCHITECTURE,
    SCRATCH_PREFIX,
    assert_model_residency,
    owned_scratch_directory,
    parse_args,
    validate_captured_mistral_inputs,
    validate_mistral_architecture,
    validate_scratch_parent,
)


class _TinyResidentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=False, dtype=torch.float16)
        self.register_buffer("indices", torch.arange(3))


class Stage1MemorySmokeHelperTest(unittest.TestCase):
    def test_scratch_argument_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--scratch-parent", directory])

        self.assertEqual(args.scratch_parent, Path(directory))
        self.assertEqual(args.device_index, 0)

    def test_owned_scratch_is_fresh_and_removes_only_its_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            sentinel = parent / "keep.txt"
            sentinel.write_text("preserve", encoding="utf-8")

            with owned_scratch_directory(parent) as scratch:
                self.assertEqual(scratch.parent, parent.resolve())
                self.assertTrue(scratch.name.startswith(SCRATCH_PREFIX))
                self.assertNotEqual(scratch, sentinel)
                (scratch / "owned.txt").write_text("temporary", encoding="utf-8")
                owned_path = scratch

            self.assertFalse(owned_path.exists())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_missing_scratch_parent_is_rejected_without_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaises(NotADirectoryError):
                validate_scratch_parent(missing)
            self.assertFalse(missing.exists())

    def test_cpu_model_residency_and_fp16_are_checked(self) -> None:
        model = _TinyResidentModel()
        summary = assert_model_residency(model, torch.device("cpu"))

        self.assertEqual(summary["parameters"]["numel"], 12)
        self.assertEqual(summary["parameters"]["bytes_by_device"], {"cpu": 24})
        self.assertEqual(summary["buffers"]["bytes_by_device"], {"cpu": 24})

        model.proj = nn.Linear(4, 3, bias=False, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "not uniformly FP16"):
            assert_model_residency(model, torch.device("cpu"))

    def test_exact_mistral_architecture_validation(self) -> None:
        config = SimpleNamespace(
            **EXPECTED_ARCHITECTURE,
            max_position_embeddings=32_768,
            sliding_window=None,
        )
        observed = validate_mistral_architecture(config)
        self.assertEqual(observed["hidden_size"], 4_096)

        config.num_key_value_heads = 4
        with self.assertRaisesRegex(RuntimeError, "num_key_value_heads"):
            validate_mistral_architecture(config)

    def test_mistral_shaped_capture_validation_is_cpu_testable(self) -> None:
        sequence_length = 4
        hidden_size = 8
        head_dim = 2
        cache_position = torch.arange(sequence_length)
        position_ids = cache_position.unsqueeze(0)
        input_args = (
            torch.randn(
                1,
                sequence_length,
                hidden_size,
                dtype=torch.float16,
            ),
        )
        input_kwargs = {
            "attention_mask": None,
            "position_ids": position_ids,
            "past_key_values": None,
            "use_cache": False,
            "cache_position": cache_position,
            "position_embeddings": (
                torch.randn(
                    1,
                    sequence_length,
                    head_dim,
                    dtype=torch.float16,
                ),
                torch.randn(
                    1,
                    sequence_length,
                    head_dim,
                    dtype=torch.float16,
                ),
            ),
        }

        summary = validate_captured_mistral_inputs(
            input_args,
            input_kwargs,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            head_dim=head_dim,
        )
        self.assertEqual(summary["hidden_states"]["shape"], [1, 4, 8])
        self.assertEqual(summary["position_embeddings"][0]["shape"], [1, 4, 2])

        input_kwargs["use_cache"] = True
        with self.assertRaisesRegex(RuntimeError, "disabled KV cache"):
            validate_captured_mistral_inputs(
                input_args,
                input_kwargs,
                sequence_length=sequence_length,
                hidden_size=hidden_size,
                head_dim=head_dim,
            )


if __name__ == "__main__":
    unittest.main()
