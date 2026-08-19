import hashlib
import os
from os import PathLike
from typing import Any, Union

import torch


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
