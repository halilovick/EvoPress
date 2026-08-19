import unittest
from pathlib import Path
from unittest import mock

from scripts.run_apples_to_apples import (
    MIN_PREPARE_DB_FREE_INODES,
    prepare_db_storage_preflight,
)


def _storage(*, device_id: int, free_inodes: int):
    return {
        "requested_path": "/fake",
        "existing_anchor": "/fake",
        "device_id": device_id,
        "filesystem_type": "ceph",
        "write_fsync_probe_passed": True,
        "free_bytes": 300 * 1024**3,
        "total_inodes": 734_000_000,
        "free_inodes": free_inodes,
    }


class Stage1StoragePreflightTest(unittest.TestCase):
    def test_unknown_free_inodes_minus_one_is_accepted(self):
        snapshots = [
            _storage(device_id=1, free_inodes=-1),
            _storage(device_id=2, free_inodes=-1),
        ]

        with mock.patch(
            "scripts.run_apples_to_apples._storage_snapshot",
            side_effect=snapshots,
        ):
            result = prepare_db_storage_preflight(
                Path("/fake/database/Mistral-7B-v0.3/3bit"),
                Path("/fake/cache"),
            )

        self.assertEqual(result["database"]["free_inodes"], -1)
        self.assertEqual(result["activation_cache"]["free_inodes"], -1)

    def test_known_low_free_inode_count_is_still_rejected(self):
        with mock.patch(
            "scripts.run_apples_to_apples._storage_snapshot",
            return_value=_storage(
                device_id=1,
                free_inodes=MIN_PREPARE_DB_FREE_INODES - 1,
            ),
        ):
            with self.assertRaisesRegex(OSError, "Insufficient free inodes"):
                prepare_db_storage_preflight(
                    Path("/fake/database/Mistral-7B-v0.3/3bit"),
                    None,
                )


if __name__ == "__main__":
    unittest.main()
