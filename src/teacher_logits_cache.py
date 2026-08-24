"""Lossless disk-backed storage for dense teacher logits."""

from __future__ import annotations

import atexit
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from src.io_utils import torch_save
from src.memory_utils import release_cpu_memory


@dataclass(frozen=True)
class DiskTensorRef:
    """Lazy reference to one serialized tensor with deferred indexing."""

    path: Path
    index_ops: tuple[Any, ...] = ()

    def __getitem__(self, item: Any) -> "DiskTensorRef":
        # Preserve the existing target_logits[i][:, :n] interface without
        # materializing the tensor in CPU RAM.
        return DiskTensorRef(
            self.path,
            self.index_ops + (item,),
        )

    def load(self) -> torch.Tensor:
        # mmap keeps the serialized tensor file-backed. Only the requested
        # pages/slice need to become resident before transfer to the GPU.
        tensor = torch.load(
            self.path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Expected Tensor in teacher-logit cache entry {self.path}, "
                f"got {type(tensor).__name__}."
            )

        for item in self.index_ops:
            tensor = tensor[item]

        return tensor

    def evict_file_cache(self) -> None:
        """Hint that cached pages for this tensor file are no longer needed."""

        if not (
            hasattr(os, "posix_fadvise")
            and hasattr(os, "POSIX_FADV_DONTNEED")
        ):
            return

        file_descriptor = None
        try:
            file_descriptor = os.open(self.path, os.O_RDONLY)
            os.posix_fadvise(
                file_descriptor,
                0,
                0,
                os.POSIX_FADV_DONTNEED,
            )
        except OSError:
            # Some overlay/network filesystems may not support the hint.
            pass
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)


def materialize_tensor_reference(value: Any) -> Any:
    """Materialize a lazy disk tensor reference; pass ordinary values through."""

    if isinstance(value, DiskTensorRef):
        return value.load()
    return value


def evict_tensor_reference_file_cache(value: Any) -> None:
    """Evict file-backed pages for a materialized lazy tensor reference."""

    if isinstance(value, DiskTensorRef):
        value.evict_file_cache()


class DiskTensorCache(Sequence[DiskTensorRef]):
    """Append-only lossless disk cache for large tensors.

    Teacher logits are serialized exactly in their produced dtype. The Python
    process retains only lightweight file references, rather than all logits
    in anonymous CPU memory.

    Temporary caches default to /tmp, but the parent can be overridden with
    EVOPRESS_TEACHER_LOGITS_CACHE_PARENT.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        drop_file_cache: bool = True,
        cleanup_on_exit: bool = False,
    ) -> None:
        self.root = Path(root)
        self.drop_file_cache = drop_file_cache
        self._refs: list[DiskTensorRef] = []
        self._payload_bytes = 0
        self._cleaned = False

        if self.root.exists():
            raise FileExistsError(
                f"Refusing to reuse existing teacher-logit cache: {self.root}"
            )

        self.root.mkdir(parents=True, exist_ok=False)

        if cleanup_on_exit:
            atexit.register(self.cleanup)

    @classmethod
    def temporary(
        cls,
        *,
        prefix: str = "evopress-teacher-logits",
        parent: str | os.PathLike[str] | None = None,
        drop_file_cache: bool = True,
    ) -> "DiskTensorCache":
        cache_parent = Path(
            parent
            or os.environ.get(
                "EVOPRESS_TEACHER_LOGITS_CACHE_PARENT",
                "/tmp",
            )
        )
        cache_parent.mkdir(parents=True, exist_ok=True)

        root = cache_parent / (
            f"{prefix}-{os.getpid()}-{uuid.uuid4().hex}"
        )

        return cls(
            root,
            drop_file_cache=drop_file_cache,
            cleanup_on_exit=True,
        )

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(
        self,
        index: int | slice,
    ) -> DiskTensorRef | list[DiskTensorRef]:
        return self._refs[index]

    @property
    def payload_bytes(self) -> int:
        return self._payload_bytes

    def append(self, tensor: torch.Tensor) -> DiskTensorRef:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "DiskTensorCache only stores tensors, got "
                f"{type(tensor).__name__}."
            )

        path = self.root / f"{len(self._refs):08d}.pth"

        # Only one full teacher-logit tensor is resident in anonymous CPU RAM
        # during serialization. It is released immediately after the save.
        cpu_tensor = tensor.detach().to(device="cpu").contiguous()

        torch_save(
            cpu_tensor,
            path,
            drop_file_cache=self.drop_file_cache,
        )

        ref = DiskTensorRef(path)
        self._refs.append(ref)
        self._payload_bytes += path.stat().st_size

        del cpu_tensor
        release_cpu_memory()

        return ref

    def cleanup(self) -> None:
        if self._cleaned:
            return

        shutil.rmtree(self.root, ignore_errors=True)
        self._cleaned = True
