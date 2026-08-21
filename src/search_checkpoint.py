"""Atomic generation-level checkpoints for long-running searches."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    """Capture all RNGs used by EvoPress search proposal/selection logic."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - set(state)
    if missing:
        raise ValueError(
            f"Checkpoint RNG state is missing fields: {sorted(missing)}"
        )

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable."
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "CUDA device count differs from checkpoint: "
                f"checkpoint={len(cuda_states)}, "
                f"current={torch.cuda.device_count()}."
            )
        torch.cuda.set_rng_state_all(cuda_states)


def save_search_checkpoint(
    path: str | os.PathLike[str],
    *,
    search_type: str,
    completed_generation: int,
    identity: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Path:
    """Atomically persist one completed-generation search state."""

    if completed_generation < 0:
        raise ValueError("completed_generation must be non-negative.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "search_type": search_type,
        "completed_generation": int(completed_generation),
        "identity": dict(identity),
        "state": dict(state),
        "rng_state": capture_rng_state(),
    }

    temporary_path = output_path.parent / (
        f".{output_path.name}.{os.getpid()}.tmp"
    )

    if temporary_path.exists():
        raise FileExistsError(
            f"Checkpoint temporary path already exists: {temporary_path}"
        )

    try:
        with temporary_path.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, output_path)

        # Persist the directory entry as well when supported.
        try:
            directory_fd = os.open(
                output_path.parent,
                os.O_RDONLY,
            )
        except OSError:
            directory_fd = None

        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return output_path


def load_search_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_search_type: str | None = None,
) -> dict[str, Any]:
    """Load and structurally validate a trusted local search checkpoint."""

    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Search checkpoint does not exist: {checkpoint_path}"
        )

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise ValueError("Search checkpoint must contain a dictionary.")

    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported search checkpoint schema: "
            f"{payload.get('schema_version')!r}"
        )

    required = {
        "search_type",
        "completed_generation",
        "identity",
        "state",
        "rng_state",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"Search checkpoint is missing fields: {sorted(missing)}"
        )

    if (
        expected_search_type is not None
        and payload["search_type"] != expected_search_type
    ):
        raise ValueError(
            "Search checkpoint type mismatch: "
            f"expected={expected_search_type!r}, "
            f"actual={payload['search_type']!r}."
        )

    return payload


def validate_checkpoint_identity(
    checkpoint: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
) -> None:
    """Refuse resume when any trajectory-defining setting differs."""

    actual = checkpoint.get("identity")

    if not isinstance(actual, Mapping):
        raise ValueError("Search checkpoint identity is invalid.")

    expected = dict(expected_identity)
    actual = dict(actual)

    if actual == expected:
        return

    mismatches = {}
    for key in sorted(set(actual) | set(expected)):
        if actual.get(key) != expected.get(key):
            mismatches[key] = {
                "checkpoint": actual.get(key),
                "current": expected.get(key),
            }

    raise ValueError(
        "Checkpoint does not match the current search configuration: "
        f"{mismatches}"
    )
