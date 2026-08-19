"""Small process/cgroup memory helpers used by long-running GPU jobs."""

from __future__ import annotations

import ctypes
import gc
import os
from pathlib import Path
from typing import Any


def _read_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def cgroup_memory_snapshot() -> dict[str, Any]:
    """Return cgroup-v2 memory counters when exposed by the runtime."""

    root = Path("/sys/fs/cgroup")
    snapshot: dict[str, Any] = {
        "limit_bytes": _read_integer(root / "memory.max"),
        "current_bytes": _read_integer(root / "memory.current"),
        "peak_bytes": _read_integer(root / "memory.peak"),
    }
    events_path = root / "memory.events"
    events: dict[str, int] = {}
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            events[parts[0]] = int(parts[1])
        except ValueError:
            continue
    snapshot["events"] = events
    return snapshot


def release_cpu_memory() -> None:
    """Collect Python garbage and return free glibc arenas when available."""

    gc.collect()
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
    except (AttributeError, OSError):
        return
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)
