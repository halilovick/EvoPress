import hashlib
import os
from os import PathLike
from typing import Any, Union

import torch


def drop_file_cache_for_path(path: Union[str, PathLike]) -> None:
    """Best-effort eviction of one file from the Linux page cache."""
    if not (
        hasattr(os, "posix_fadvise")
        and hasattr(os, "POSIX_FADV_DONTNEED")
    ):
        return

    fd = None
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(
            fd,
            0,
            0,
            os.POSIX_FADV_DONTNEED,
        )
    except OSError:
        # Overlay/network filesystems may not support the hint.
        pass
    finally:
        if fd is not None:
            os.close(fd)


def torch_load_tensor(
    path: Union[str, PathLike],
    *,
    device=None,
    dtype=None,
    drop_file_cache: bool = False,
) -> torch.Tensor:
    """Load one tensor without retaining its serialized file in page cache.

    The file is memory-mapped on CPU, copied into independent destination
    storage, and the mmap-backed source is released before the optional
    POSIX_FADV_DONTNEED hint.
    """
    source = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(source, torch.Tensor):
        raise TypeError(
            f"Expected Tensor in {path}, got {type(source).__name__}."
        )

    target_device = source.device if device is None else device
    target_dtype = source.dtype if dtype is None else dtype

    # copy=True is important even for CPU destinations: the returned tensor
    # must not keep the mmap-backed serialized storage alive.
    result = source.to(
        device=target_device,
        dtype=target_dtype,
        copy=True,
    )

    del source

    if drop_file_cache:
        drop_file_cache_for_path(path)

    return result


class _HashingWriter:
    def __init__(self, handle, digest) -> None:
        self.handle = handle
        self.digest = digest

    def write(self, value):
        self.digest.update(value)
        return self.handle.write(value)

    def __getattr__(self, name):
        return getattr(self.handle, name)


def torch_save(
    obj: Any,
    path: Union[str, PathLike],
    drop_file_cache: bool = False,
    compute_sha256: bool = False,
) -> str | None:
    """Save a PyTorch object and optionally evict the written file from page cache."""
    if not drop_file_cache and not compute_sha256:
        torch.save(obj, path)
        return None

    digest = hashlib.sha256() if compute_sha256 else None
    with open(path, "wb") as handle:
        destination = _HashingWriter(handle, digest) if digest is not None else handle
        torch.save(obj, destination)
        handle.flush()
        os.fsync(handle.fileno())

        if drop_file_cache and hasattr(os, "posix_fadvise") and hasattr(
            os, "POSIX_FADV_DONTNEED"
        ):
            try:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                # Some network and overlay filesystems do not support this hint.
                pass
    return digest.hexdigest() if digest is not None else None
