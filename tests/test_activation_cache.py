import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.activation_cache import DiskActivationCache
from src.quantizer import Quantizer


class _TinyBlock(nn.Module):
    """Decoder-like block that consumes the Mistral-shaped cached kwargs."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width, bias=True)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        del past_key_values, use_cache, cache_position
        position_term = position_ids.to(hidden_states.dtype).unsqueeze(-1) * 0.001
        mask_term = attention_mask.to(hidden_states.dtype).unsqueeze(-1) * 0.0001
        cosine, sine = position_embeddings
        rope_term = (cosine[..., :1] + sine[..., :1]) * 0.0001
        output = torch.tanh(
            self.proj(hidden_states) + position_term + mask_term + rope_term
        )
        # Exercise Quantizer's maybe_first_element tuple-output path.
        return output, {"unused_auxiliary_output": True}


class _TinyBackbone(nn.Module):
    def __init__(self, width: int, num_blocks: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, width)
        self.layers = nn.ModuleList(
            [_TinyBlock(width) for _ in range(num_blocks)]
        )


class _TinyCausalLM(nn.Module):
    def __init__(self, width: int = 4, num_blocks: int = 2) -> None:
        super().__init__()
        self.model = _TinyBackbone(width, num_blocks)
        self.config = SimpleNamespace(use_cache=True)

    def forward(self, input_ids):
        hidden_states = self.model.embed_tokens(input_ids)
        cache_position = torch.arange(
            hidden_states.shape[1], device=hidden_states.device
        )
        position_ids = cache_position.unsqueeze(0)
        angle = position_ids.to(hidden_states.dtype).unsqueeze(-1)
        position_embeddings = (
            torch.cos(angle).expand(-1, -1, 2),
            torch.sin(angle).expand(-1, -1, 2),
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        for layer in self.model.layers:
            output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=self.config.use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = output[0] if isinstance(output, tuple) else output
        return hidden_states


def _calibration_data():
    return [
        ([], {"input_ids": torch.tensor([[1, 2, 3, 4]])}),
        ([], {"input_ids": torch.tensor([[4, 3, 2]])}),
        ([], {"input_ids": torch.tensor([[5, 6]])}),
    ]


def _make_quantizer(model, data, save_dir, activation_cache_dir=None):
    return Quantizer(
        model,
        data,
        quantizable_modules=r".*proj$",
        pre_block_modules=["model.embed_tokens"],
        block_modules="model.layers",
        save_dir=save_dir,
        obq_kwargs={
            "rel_damp": 0.1,
            "block_size": 2,
            "perchannel": True,
            "group_size": 2,
            "sym": False,
            "act_order": False,
        },
        device=torch.device("cpu"),
        cpu_offload_modules=False,
        cpu_offload_activations=True,
        activation_cache_dir=activation_cache_dir,
        drop_saved_file_cache=False,
        verbose=False,
    )


class DiskActivationCacheTest(unittest.TestCase):
    def test_round_trip_is_ordered_and_preserves_nested_mistral_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = DiskActivationCache(Path(directory) / "cache")
            expected = []
            for marker, sequence_length in ((7, 3), (2, 1), (11, 4)):
                hidden_states = (
                    torch.arange(sequence_length * 8, dtype=torch.float16)
                    .reshape(sequence_length, 8)
                    .transpose(0, 1)
                )
                cache_position = torch.arange(sequence_length, dtype=torch.int64)
                position_ids = cache_position.unsqueeze(0)
                cosine = torch.arange(
                    sequence_length * 2, dtype=torch.float16
                ).reshape(1, sequence_length, 2)
                sine = -cosine
                input_args = (hidden_states, marker)
                input_kwargs = {
                    "attention_mask": None,
                    "position_ids": position_ids,
                    "past_key_values": None,
                    "use_cache": False,
                    "cache_position": cache_position,
                    "position_embeddings": (cosine, sine),
                    "nested": {"marker": marker, "flags": [True, False]},
                }
                expected.append((input_args, input_kwargs))
                self.assertEqual(
                    cache.append(input_args, input_kwargs), len(expected) - 1
                )

            cache.finish_collection()

            observed_markers = []
            for index, input_args, input_kwargs in Quantizer._input_items(
                cache, [], []
            ):
                expected_args, expected_kwargs = expected[index]
                observed_markers.append(input_args[1])

                self.assertIsInstance(input_args, tuple)
                self.assertIsInstance(input_kwargs, dict)
                self.assertIsInstance(input_kwargs["position_embeddings"], tuple)
                self.assertEqual(input_args[0].stride(), expected_args[0].stride())
                self.assertTrue(torch.equal(input_args[0], expected_args[0]))
                self.assertEqual(input_args[0].dtype, expected_args[0].dtype)
                self.assertEqual(input_args[0].device.type, "cpu")
                self.assertIsNone(input_kwargs["attention_mask"])
                self.assertIsNone(input_kwargs["past_key_values"])
                self.assertFalse(input_kwargs["use_cache"])
                self.assertTrue(
                    torch.equal(
                        input_kwargs["position_ids"],
                        expected_kwargs["position_ids"],
                    )
                )
                self.assertTrue(
                    torch.equal(
                        input_kwargs["cache_position"],
                        expected_kwargs["cache_position"],
                    )
                )
                for observed, expected_tensor in zip(
                    input_kwargs["position_embeddings"],
                    expected_kwargs["position_embeddings"],
                ):
                    self.assertTrue(torch.equal(observed, expected_tensor))
                self.assertEqual(input_kwargs["nested"], expected_kwargs["nested"])

            self.assertEqual(observed_markers, [7, 2, 11])
            self.assertEqual(
                [path.name for path in sorted(cache.samples_dir.iterdir())],
                ["00000000.pth", "00000001.pth", "00000002.pth"],
            )
            manifest = json.loads(
                (cache.root / cache.MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "processing")
            self.assertEqual(manifest["sample_count"], 3)

    def test_nonempty_cache_path_is_refused_without_touching_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory) / "cache"
            cache_root.mkdir()
            sentinel = cache_root / "existing.txt"
            sentinel.write_text("do not overwrite", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                DiskActivationCache(cache_root)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not overwrite")
            self.assertEqual(list(cache_root.iterdir()), [sentinel])

    def test_existing_empty_cache_path_is_also_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory) / "cache"
            cache_root.mkdir()

            with self.assertRaises(FileExistsError):
                DiskActivationCache(cache_root)

            self.assertEqual(list(cache_root.iterdir()), [])

    def test_hidden_state_replacement_preserves_all_other_inputs(self):
        replacement = torch.full((1, 3, 4), 9.0)
        position_ids = torch.arange(3).unsqueeze(0)
        positional_args = (torch.zeros_like(replacement), "static-positional")
        positional_kwargs = {"position_ids": position_ids, "use_cache": False}

        updated_positional_args, updated_positional_kwargs = (
            Quantizer._replace_hidden_state(
                positional_args, positional_kwargs, replacement
            )
        )
        self.assertTrue(torch.equal(updated_positional_args[0], replacement))
        self.assertEqual(updated_positional_args[1], "static-positional")
        self.assertTrue(
            torch.equal(updated_positional_kwargs["position_ids"], position_ids)
        )
        self.assertFalse(updated_positional_kwargs["use_cache"])

        keyword_args = ()
        keyword_kwargs = {
            "hidden_states": torch.zeros_like(replacement),
            "position_ids": position_ids,
            "position_embeddings": (
                torch.ones(1, 3, 2),
                -torch.ones(1, 3, 2),
            ),
        }
        updated_keyword_args, updated_keyword_kwargs = (
            Quantizer._replace_hidden_state(
                keyword_args, keyword_kwargs, replacement
            )
        )
        self.assertEqual(updated_keyword_args, ())
        self.assertTrue(
            torch.equal(updated_keyword_kwargs["hidden_states"], replacement)
        )
        self.assertTrue(
            torch.equal(updated_keyword_kwargs["position_ids"], position_ids)
        )
        self.assertTrue(
            torch.equal(
                updated_keyword_kwargs["position_embeddings"][0],
                keyword_kwargs["position_embeddings"][0],
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            cache = DiskActivationCache(Path(directory) / "cache")
            cache.append(positional_args, positional_kwargs)
            cache.finish_collection()
            cache.replace(0, updated_positional_args, updated_positional_kwargs)
            loaded_args, loaded_kwargs = cache.load(0)
            self.assertTrue(torch.equal(loaded_args[0], replacement))
            self.assertEqual(loaded_args[1], "static-positional")
            self.assertTrue(torch.equal(loaded_kwargs["position_ids"], position_ids))


class DiskBackedQuantizerEquivalenceTest(unittest.TestCase):
    def test_real_fastobq_disk_cache_matches_in_memory_bit_exactly(self):
        torch.manual_seed(22)
        base_model = _TinyCausalLM()
        in_memory_model = copy.deepcopy(base_model)
        disk_backed_model = copy.deepcopy(base_model)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            in_memory_dir = root / "in-memory-weights"
            disk_backed_dir = root / "disk-backed-weights"
            activation_cache_dir = root / "activation-cache"

            _make_quantizer(
                in_memory_model,
                _calibration_data(),
                in_memory_dir,
            ).quantize([2, 3], calibration_bitwidth=3)
            _make_quantizer(
                disk_backed_model,
                _calibration_data(),
                disk_backed_dir,
                activation_cache_dir=activation_cache_dir,
            ).quantize([2, 3], calibration_bitwidth=3)

            for block_index in range(2):
                module_name = f"model.layers.{block_index}.proj"
                for bitwidth in (2, 3):
                    in_memory_weight = torch.load(
                        in_memory_dir / module_name / f"{bitwidth}.pth",
                        map_location="cpu",
                        weights_only=True,
                    )
                    disk_backed_weight = torch.load(
                        disk_backed_dir / module_name / f"{bitwidth}.pth",
                        map_location="cpu",
                        weights_only=True,
                    )
                    self.assertEqual(in_memory_weight.dtype, disk_backed_weight.dtype)
                    self.assertEqual(in_memory_weight.shape, disk_backed_weight.shape)
                    self.assertTrue(
                        torch.equal(in_memory_weight, disk_backed_weight),
                        msg=f"Mismatch in {module_name} at {bitwidth} bits.",
                    )

            for _, input_kwargs in _calibration_data():
                in_memory_output = in_memory_model(**input_kwargs)
                disk_backed_output = disk_backed_model(**input_kwargs)
                self.assertTrue(torch.equal(in_memory_output, disk_backed_output))

            self.assertTrue(in_memory_model.config.use_cache)
            self.assertTrue(disk_backed_model.config.use_cache)
            cache_manifest = json.loads(
                (
                    activation_cache_dir
                    / DiskActivationCache.MANIFEST_NAME
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(cache_manifest["status"], "complete")
            self.assertEqual(cache_manifest["sample_count"], 3)
            self.assertEqual(cache_manifest["completed_blocks"], 2)
            self.assertEqual(cache_manifest["total_blocks"], 2)


if __name__ == "__main__":
    unittest.main()
