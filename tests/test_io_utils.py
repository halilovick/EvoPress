import hashlib
import os
import tempfile
import unittest
from unittest import mock

import torch

from src.io_utils import torch_load_tensor, torch_save


class TorchSaveTest(unittest.TestCase):
    def test_round_trip_without_cache_eviction(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tensor.pth")
            torch_save(tensor, path)

            loaded = torch.load(path)

        self.assertTrue(torch.equal(tensor, loaded))

    def test_cache_eviction_path_preserves_saved_tensor(self):
        tensor = torch.tensor([4.0, 5.0])

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tensor.pth")
            with mock.patch("src.io_utils.os.fsync") as fsync:
                torch_save(tensor, path, drop_file_cache=True)

            loaded = torch.load(path)

        fsync.assert_called_once()
        self.assertTrue(torch.equal(tensor, loaded))

    def test_optional_hash_matches_exact_saved_file(self):
        tensor = torch.tensor([6.0, 7.0], dtype=torch.float16)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tensor.pth")
            digest = torch_save(
                tensor,
                path,
                drop_file_cache=True,
                compute_sha256=True,
            )
            with open(path, "rb") as handle:
                expected = hashlib.sha256(handle.read()).hexdigest()

        self.assertEqual(digest, expected)

    def test_tensor_loader_preserves_tensor_exactly(self):
        tensor = torch.randn(8, 16, dtype=torch.float16)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tensor.pth")
            torch.save(tensor, path)

            loaded = torch_load_tensor(
                path,
                device="cpu",
                dtype=torch.float16,
            )

        self.assertTrue(torch.equal(tensor, loaded))

    def test_tensor_loader_evicts_read_file_cache(self):
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float16)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tensor.pth")
            torch.save(tensor, path)

            with mock.patch(
                "src.io_utils.drop_file_cache_for_path"
            ) as drop_file_cache:
                loaded = torch_load_tensor(
                    path,
                    device="cpu",
                    dtype=torch.float16,
                    drop_file_cache=True,
                )

        drop_file_cache.assert_called_once_with(path)
        self.assertTrue(torch.equal(tensor, loaded))


if __name__ == "__main__":
    unittest.main()
