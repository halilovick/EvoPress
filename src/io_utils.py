import os
from os import PathLike
from typing import Any, Union

import torch


def torch_save(
    obj: Any,
    path: Union[str, PathLike],
    drop_file_cache: bool = False,
) -> None:
    """Save a PyTorch object and optionally evict the written file from page cache."""
    if not drop_file_cache:
        torch.save(obj, path)
        return

    with open(path, "wb") as handle:
        torch.save(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())

        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                # Some network and overlay filesystems do not support this hint.
                pass
