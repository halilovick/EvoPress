import os
import tempfile
import unittest
from unittest import mock

import torch

from src.io_utils import torch_save


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


if __name__ == "__main__":
    unittest.main()
