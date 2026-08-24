import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import torch

from src.metrics import compute_kl_div
from src.teacher_logits_cache import (
    DiskTensorCache,
    DiskTensorRef,
)


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.projection = torch.nn.Linear(8, 32)

    def forward(self, input_ids):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.projection(hidden))


class TeacherLogitsCacheTest(unittest.TestCase):
    def test_round_trip_and_deferred_slice_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "teacher-cache"
            cache = DiskTensorCache(root, drop_file_cache=False)

            tensor = torch.randn(
                1,
                12,
                32,
                dtype=torch.float16,
            )
            ref = cache.append(tensor)

            self.assertEqual(len(cache), 1)
            self.assertIsInstance(cache[0], DiskTensorRef)
            self.assertTrue(torch.equal(cache[0].load(), tensor))
            self.assertTrue(
                torch.equal(
                    ref[:, :5].load(),
                    tensor[:, :5],
                )
            )
            self.assertGreater(cache.payload_bytes, 0)

            cache.cleanup()
            self.assertFalse(root.exists())

    def test_disk_backed_dense_kl_matches_in_memory_dense_kl(self):
        torch.manual_seed(0)

        teacher = TinyLM()
        student = TinyLM()
        student.load_state_dict(teacher.state_dict())

        # Make the student non-identical so the test compares a real,
        # non-zero KL value rather than only the zero case.
        with torch.no_grad():
            student.projection.weight[0].add_(0.1)

        calibration_data = [
            torch.randint(0, 32, (1, 7)),
            torch.randint(0, 32, (1, 5)),
            torch.randint(0, 32, (1, 9)),
        ]

        with torch.no_grad():
            in_memory_targets = [
                teacher(sample).logits.cpu()
                for sample in calibration_data
            ]

        in_memory_kl = compute_kl_div(
            student,
            calibration_data,
            in_memory_targets,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = DiskTensorCache(
                Path(temporary_directory) / "teacher-cache",
                drop_file_cache=False,
            )
            for tensor in in_memory_targets:
                cache.append(tensor)

            with mock.patch.object(
                DiskTensorRef,
                "evict_file_cache",
                autospec=True,
            ) as evict_file_cache:
                disk_backed_kl = compute_kl_div(
                    student,
                    calibration_data,
                    cache,
                )

            self.assertEqual(
                evict_file_cache.call_count,
                len(calibration_data),
            )

        self.assertGreater(in_memory_kl, 0.0)
        self.assertAlmostEqual(
            disk_backed_kl,
            in_memory_kl,
            places=7,
        )

    def test_evict_file_cache_uses_posix_dontneed_when_available(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tensor.pth"
            torch.save(torch.ones(4), path)
            ref = DiskTensorRef(path)

            with (
                mock.patch.object(
                    os,
                    "posix_fadvise",
                    create=True,
                ) as posix_fadvise,
                mock.patch.object(
                    os,
                    "POSIX_FADV_DONTNEED",
                    4,
                    create=True,
                ),
            ):
                ref.evict_file_cache()

            self.assertEqual(posix_fadvise.call_count, 1)
            _, offset, length, advice = posix_fadvise.call_args.args
            self.assertEqual(offset, 0)
            self.assertEqual(length, 0)
            self.assertEqual(advice, 4)


if __name__ == "__main__":
    unittest.main()
