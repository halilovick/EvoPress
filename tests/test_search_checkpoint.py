import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.search_checkpoint import (
    load_search_checkpoint,
    restore_rng_state,
    save_search_checkpoint,
    validate_checkpoint_identity,
)


class SearchCheckpointTest(unittest.TestCase):
    def test_resume_restores_all_rng_sequences(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search_checkpoint.pt"

            save_search_checkpoint(
                path,
                search_type="quant_only",
                completed_generation=7,
                identity={"seed": 123, "offspring": 4},
                state={"parent": [[3, 3]], "train_fitness": 0.5},
            )

            expected_python = [random.random() for _ in range(5)]
            expected_numpy = np.random.random(5)
            expected_torch = torch.rand(5)

            # Move every RNG somewhere else.
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)

            checkpoint = load_search_checkpoint(
                path,
                expected_search_type="quant_only",
            )
            restore_rng_state(checkpoint["rng_state"])

            self.assertEqual(
                [random.random() for _ in range(5)],
                expected_python,
            )
            np.testing.assert_array_equal(
                np.random.random(5),
                expected_numpy,
            )
            self.assertTrue(
                torch.equal(
                    torch.rand(5),
                    expected_torch,
                )
            )

    def test_atomic_overwrite_keeps_latest_completed_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search_checkpoint.pt"
            identity = {"seed": 0}

            save_search_checkpoint(
                path,
                search_type="quant_only",
                completed_generation=1,
                identity=identity,
                state={"value": 1},
            )
            save_search_checkpoint(
                path,
                search_type="quant_only",
                completed_generation=2,
                identity=identity,
                state={"value": 2},
            )

            checkpoint = load_search_checkpoint(path)
            self.assertEqual(
                checkpoint["completed_generation"],
                2,
            )
            self.assertEqual(
                checkpoint["state"]["value"],
                2,
            )

    def test_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search_checkpoint.pt"

            save_search_checkpoint(
                path,
                search_type="quant_only",
                completed_generation=1,
                identity={"seed": 0, "offspring": 128},
                state={},
            )
            checkpoint = load_search_checkpoint(path)

            with self.assertRaises(ValueError):
                validate_checkpoint_identity(
                    checkpoint,
                    {"seed": 1, "offspring": 128},
                )


if __name__ == "__main__":
    unittest.main()
