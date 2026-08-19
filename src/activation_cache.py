"""Lossless disk-backed storage for block calibration activations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from src.common_utils import to
from src.io_utils import torch_save


class DiskActivationCache:
    """Store one block-input payload per file and replace it atomically.

    The cache deliberately has no resume mode. Any existing path is rejected so
    a failed run can never be mistaken for a complete calibration state.
    """

    MANIFEST_NAME = "activation_cache_manifest.json"

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        drop_file_cache: bool = False,
        attempt_id: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.samples_dir = self.root / "samples"
        self.drop_file_cache = drop_file_cache
        self.attempt_id = attempt_id
        self._count = 0
        self._current_size_bytes = 0
        self._bytes_read_cumulative = 0
        self._bytes_written_cumulative = 0

        if self.root.exists():
            raise FileExistsError(
                f"Refusing to reuse existing activation cache: {self.root}"
            )
        self.samples_dir.mkdir(parents=True, exist_ok=False)
        self._write_manifest(status="collecting", completed_blocks=0)

    def __len__(self) -> int:
        return self._count

    def _sample_path(self, index: int) -> Path:
        if index < 0 or index >= self._count:
            raise IndexError(
                f"Activation-cache index {index} is outside [0, {self._count})."
            )
        return self.samples_dir / f"{index:08d}.pth"

    @staticmethod
    def _cpu_payload(input_args: Any, input_kwargs: Any) -> dict[str, Any]:
        return {
            "input_args": to(input_args, device="cpu"),
            "input_kwargs": to(input_kwargs, device="cpu"),
        }

    def append(self, input_args: Any, input_kwargs: Any) -> int:
        index = self._count
        path = self.samples_dir / f"{index:08d}.pth"
        torch_save(
            self._cpu_payload(input_args, input_kwargs),
            path,
            drop_file_cache=self.drop_file_cache,
        )
        size_bytes = path.stat().st_size
        self._current_size_bytes += size_bytes
        self._bytes_written_cumulative += size_bytes
        self._count += 1
        return index

    def load(self, index: int) -> tuple[Any, Any]:
        path = self._sample_path(index)
        self._bytes_read_cumulative += path.stat().st_size
        with path.open("rb") as handle:
            payload = torch.load(handle, map_location="cpu", weights_only=False)
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                try:
                    os.posix_fadvise(
                        handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                    )
                except OSError:
                    pass
        if not isinstance(payload, dict) or set(payload) != {
            "input_args",
            "input_kwargs",
        }:
            raise ValueError(f"Invalid activation-cache payload: {path}")
        return payload["input_args"], payload["input_kwargs"]

    def replace(self, index: int, input_args: Any, input_kwargs: Any) -> None:
        path = self._sample_path(index)
        old_size_bytes = path.stat().st_size
        temporary_path = self.samples_dir / f".{index:08d}.{os.getpid()}.tmp"
        if temporary_path.exists():
            raise FileExistsError(
                f"Activation-cache temporary path already exists: {temporary_path}"
            )
        try:
            torch_save(
                self._cpu_payload(input_args, input_kwargs),
                temporary_path,
                drop_file_cache=self.drop_file_cache,
            )
            new_size_bytes = temporary_path.stat().st_size
            os.replace(temporary_path, path)
            self._current_size_bytes += new_size_bytes - old_size_bytes
            self._bytes_written_cumulative += new_size_bytes
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def finish_collection(self) -> None:
        if self._count == 0:
            raise ValueError("Cannot finalize an empty activation cache.")
        self._write_manifest(status="processing", completed_blocks=0)

    def mark_block_complete(self, completed_blocks: int, total_blocks: int) -> None:
        if completed_blocks < 1 or completed_blocks > total_blocks:
            raise ValueError(
                f"completed_blocks must be in [1, {total_blocks}], got "
                f"{completed_blocks}."
            )
        status = "complete" if completed_blocks == total_blocks else "processing"
        self._write_manifest(
            status=status,
            completed_blocks=completed_blocks,
            total_blocks=total_blocks,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "sample_count": self._count,
            "current_size_bytes": self._current_size_bytes,
            "bytes_read_cumulative": self._bytes_read_cumulative,
            "bytes_written_cumulative": self._bytes_written_cumulative,
            "path": str(self.root),
        }

    def _write_manifest(
        self,
        *,
        status: str,
        completed_blocks: int,
        total_blocks: int | None = None,
    ) -> None:
        value = {
            "schema_version": 1,
            "attempt_id": self.attempt_id,
            "status": status,
            "sample_count": self._count,
            "current_size_bytes": self._current_size_bytes,
            "bytes_read_cumulative": self._bytes_read_cumulative,
            "bytes_written_cumulative": self._bytes_written_cumulative,
            "completed_blocks": completed_blocks,
            "total_blocks": total_blocks,
            "representation": "lossless torch serialization of block input args/kwargs",
        }
        path = self.root / self.MANIFEST_NAME
        temporary_path = self.root / f".{self.MANIFEST_NAME}.{os.getpid()}.tmp"
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
