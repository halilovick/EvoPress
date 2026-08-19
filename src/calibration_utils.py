"""Calibration partition helpers shared by reproducible database launchers."""

from __future__ import annotations

import hashlib
import struct

import torch


def calibration_token_digest(samples) -> str:
    """Hash ordered token tensors, including dtype, shape, and sample boundaries."""

    digest = hashlib.sha256()
    digest.update(b"evopress-calibration-token-digest-v1\0")
    for sample in samples:
        if not isinstance(sample, torch.Tensor):
            raise TypeError(
                "Calibration-token digest expects a sequence of torch.Tensor values."
            )
        tensor = sample.detach().to(device="cpu").contiguous()
        dtype_name = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack("<I", len(dtype_name)))
        digest.update(dtype_name)
        digest.update(struct.pack("<I", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack("<Q", int(dimension)))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()

def configured_calibration_partition(
    total_sequences: int,
    logical_shard_count: int,
    world_size: int,
    rank: int,
) -> tuple[int, int, int]:
    """Return the configured prefix size and this effective rank's slice.

    EvoPress's distributed calibration uses floor division by the configured
    worker count. A smaller physical world can reproduce the same example set
    by first retaining that exact prefix and then partitioning it across the
    effective workers.
    """

    if total_sequences < 0:
        raise ValueError("total_sequences must be non-negative.")
    if logical_shard_count < 1:
        raise ValueError("logical_shard_count must be at least 1.")
    if world_size < 1:
        raise ValueError("world_size must be at least 1.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}.")

    used_sequences = (
        total_sequences // logical_shard_count
    ) * logical_shard_count
    if used_sequences == 0:
        raise ValueError(
            "Calibration data contains fewer sequences than the configured "
            f"logical shard count ({logical_shard_count})."
        )
    if used_sequences % world_size != 0:
        raise ValueError(
            "Configured calibration prefix cannot be split equally across the "
            f"effective world size: used_sequences={used_sequences}, "
            f"world_size={world_size}."
        )

    sequences_per_rank = used_sequences // world_size
    start = rank * sequences_per_rank
    return used_sequences, start, start + sequences_per_rank
