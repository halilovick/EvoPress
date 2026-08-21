#!/usr/bin/env python3
"""Reproducible launcher for the EvoPress apples-to-apples experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "apples_to_apples"
    / "mistral7b_v03_paper_matched.json"
)
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTENTION_PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj"}
EXPECTED_MISTRAL_RECONSTRUCTION_SHAPES = {
    "q_proj": [4096, 4096],
    "k_proj": [1024, 4096],
    "v_proj": [1024, 4096],
    "o_proj": [4096, 4096],
    "gate_proj": [14336, 4096],
    "up_proj": [14336, 4096],
    "down_proj": [4096, 14336],
}
GIB = 1024**3
MIN_DATABASE_FREE_BYTES = 100 * GIB
MIN_ACTIVATION_CACHE_FREE_BYTES = 100 * GIB
MIN_COMBINED_FREE_BYTES = 200 * GIB
MIN_PREPARE_DB_FREE_INODES = 20_000
LEGACY_ACTIVATION_MEMORY_BYTES = 72 * GIB
LEGACY_MODEL_MEMORY_BYTES_PER_PROCESS = 16 * GIB
LEGACY_RUNTIME_HEADROOM_BYTES = 16 * GIB
MEMORY_BACKED_FILESYSTEMS = {"tmpfs", "ramfs"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary_path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(f"No existing ancestor for path: {path}")
        candidate = parent
    return candidate


def _storage_snapshot(path: Path) -> dict[str, Any]:
    anchor = _nearest_existing_path(path)
    usage = shutil.disk_usage(anchor)
    filesystem = os.statvfs(anchor)
    filesystem_type = _linux_filesystem_type(anchor)
    if filesystem_type in MEMORY_BACKED_FILESYSTEMS:
        raise OSError(
            "Refusing memory-backed storage for Stage 1 because its pages count "
            f"against the CPU cgroup: path={path}, anchor={anchor}, "
            f"filesystem_type={filesystem_type}."
        )
    probe_path = anchor / f".evopress-write-probe-{os.getpid()}-{uuid.uuid4().hex}"
    file_descriptor = None
    try:
        file_descriptor = os.open(
            probe_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        os.write(file_descriptor, b"evopress-storage-preflight\n")
        os.fsync(file_descriptor)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if probe_path.exists():
            probe_path.unlink()
    return {
        "requested_path": str(path),
        "existing_anchor": str(anchor),
        "device_id": int(anchor.stat().st_dev),
        "filesystem_type": filesystem_type,
        "write_fsync_probe_passed": True,
        "free_bytes": int(usage.free),
        "total_inodes": int(filesystem.f_files),
        "free_inodes": int(filesystem.f_favail),
    }


def _linux_filesystem_type(path: Path) -> str | None:
    """Return the longest matching Linux mount type, if mountinfo is available."""

    mountinfo = Path("/proc/self/mountinfo")
    try:
        lines = mountinfo.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    resolved = path.resolve()
    matches: list[tuple[int, str]] = []
    escape_sequences = {
        "\\040": " ",
        "\\011": "\t",
        "\\012": "\n",
        "\\134": "\\",
    }
    for line in lines:
        try:
            prefix, suffix = line.split(" - ", 1)
            mount_value = prefix.split()[4]
            filesystem_type = suffix.split()[0]
        except (IndexError, ValueError):
            continue
        for encoded, decoded in escape_sequences.items():
            mount_value = mount_value.replace(encoded, decoded)
        mount_path = Path(mount_value)
        if resolved == mount_path or mount_path in resolved.parents:
            matches.append((len(str(mount_path)), filesystem_type))
    return max(matches)[1] if matches else None


def prepare_db_storage_preflight(
    quant_db: Path,
    activation_cache_dir: Path | None,
) -> dict[str, Any]:
    """Fail before launch when Stage-1 output/spill capacity is clearly unsafe."""

    database_storage = _storage_snapshot(quant_db.parent)
    if (
        database_storage["total_inodes"] > 0
        and 0 <= database_storage["free_inodes"] < MIN_PREPARE_DB_FREE_INODES
    ):
        raise OSError(
            "Insufficient free inodes for quantization database generation: "
            f"{database_storage}."
        )
    result: dict[str, Any] = {
        "database": database_storage,
        "minimum_database_free_bytes": MIN_DATABASE_FREE_BYTES,
        "minimum_free_inodes": MIN_PREPARE_DB_FREE_INODES,
    }
    if activation_cache_dir is None:
        if database_storage["free_bytes"] < MIN_DATABASE_FREE_BYTES:
            raise OSError(
                "Insufficient free space for the quantization database: "
                f"required={MIN_DATABASE_FREE_BYTES}, actual={database_storage['free_bytes']}."
            )
        return result

    resolved_db = quant_db.resolve()
    resolved_cache = activation_cache_dir.resolve()
    if (
        resolved_db == resolved_cache
        or resolved_db in resolved_cache.parents
        or resolved_cache in resolved_db.parents
    ):
        raise ValueError(
            "Activation cache and quantization database must be separate, "
            f"non-nested paths: database={resolved_db}, cache={resolved_cache}."
        )
    cache_storage = _storage_snapshot(resolved_cache.parent)
    if (
        cache_storage["total_inodes"] > 0
        and 0 <= cache_storage["free_inodes"] < MIN_PREPARE_DB_FREE_INODES
    ):
        raise OSError(
            "Insufficient free inodes for the activation cache: "
            f"{cache_storage}."
        )
    same_filesystem = (
        database_storage["device_id"] == cache_storage["device_id"]
    )
    result.update(
        {
            "activation_cache": cache_storage,
            "same_filesystem": same_filesystem,
            "minimum_activation_cache_free_bytes": MIN_ACTIVATION_CACHE_FREE_BYTES,
            "minimum_combined_free_bytes": MIN_COMBINED_FREE_BYTES,
        }
    )
    if same_filesystem:
        if database_storage["free_bytes"] < MIN_COMBINED_FREE_BYTES:
            raise OSError(
                "Database and activation cache share a filesystem without the "
                f"required {MIN_COMBINED_FREE_BYTES} free bytes: {result}."
            )
    else:
        if database_storage["free_bytes"] < MIN_DATABASE_FREE_BYTES:
            raise OSError(
                "Insufficient free space for the quantization database: "
                f"required={MIN_DATABASE_FREE_BYTES}, actual={database_storage['free_bytes']}."
            )
        if cache_storage["free_bytes"] < MIN_ACTIVATION_CACHE_FREE_BYTES:
            raise OSError(
                "Insufficient free space for the activation cache: "
                f"required={MIN_ACTIVATION_CACHE_FREE_BYTES}, actual={cache_storage['free_bytes']}."
            )
    return result


def _read_cgroup_integer(path: Path) -> int | None:
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
    root = Path("/sys/fs/cgroup")
    events: dict[str, int] = {}
    try:
        event_lines = (root / "memory.events").read_text(encoding="utf-8").splitlines()
    except OSError:
        event_lines = []
    for line in event_lines:
        fields = line.split()
        if len(fields) == 2:
            try:
                events[fields[0]] = int(fields[1])
            except ValueError:
                pass
    return {
        "current_bytes": _read_cgroup_integer(root / "memory.current"),
        "limit_bytes": _read_cgroup_integer(root / "memory.max"),
        "peak_bytes": _read_cgroup_integer(root / "memory.peak"),
        "events": events,
    }


def validate_prepare_db_memory_mode(
    memory_mode: str,
    effective_processes: int,
    memory_snapshot: dict[str, Any],
) -> None:
    limit_bytes = memory_snapshot.get("limit_bytes")
    required_legacy_bytes = (
        LEGACY_ACTIVATION_MEMORY_BYTES
        + effective_processes * LEGACY_MODEL_MEMORY_BYTES_PER_PROCESS
        + LEGACY_RUNTIME_HEADROOM_BYTES
    )
    if (
        memory_mode == "legacy_cpu_offload"
        and isinstance(limit_bytes, int)
        and limit_bytes < required_legacy_bytes
    ):
        raise RuntimeError(
            "Legacy database generation retains approximately 72 GiB of shared "
            "activations plus one approximately 16 GiB model allocation per "
            "effective process and runtime headroom. The conservative aggregate "
            f"requirement for {effective_processes} process(es) is "
            f"{required_legacy_bytes} bytes, but the detected shared cgroup limit "
            f"is only {limit_bytes} bytes; use the explicit disk_activation_cache "
            "mode with a fresh spill path."
        )


def expected_search_compute(config: dict[str, Any], method: str) -> dict[str, int]:
    if method not in {"quant_only", "joint"}:
        return {"candidate_evaluations": 0, "candidate_tokens": 0}
    search = config["search"]
    if method == "quant_only":
        initial_evaluations = (
            0 if search["skip_quant_uniform_initial_evaluation"] else 1
        )
    else:
        initial_evaluations = (
            0
            if search.get("skip_joint_single_initial_evaluation", False)
            else search["joint_initial_candidates"]
        )

    stage_counts = []
    candidate_pool = search["offspring"]
    for stage_index, survivor_count in enumerate(search["survivors"]):
        if stage_index == len(search["survivors"]) - 1:
            candidate_pool += 1  # incumbent parent, excluded from generated offspring
        stage_counts.append(candidate_pool)
        candidate_pool = survivor_count
    evaluations_per_generation = sum(stage_counts)
    tokens_per_generation = sum(
        count * tokens
        for count, tokens in zip(stage_counts, search["selection_tokens"])
    )
    return {
        "candidate_evaluations": (
            initial_evaluations
            + search["generations"] * evaluations_per_generation
        ),
        "candidate_tokens": (
            initial_evaluations * search["initial_tokens"]
            + search["generations"] * tokens_per_generation
        ),
    }


def expected_mistral_module_names() -> set[str]:
    names = set()
    for layer_index in range(32):
        for projection in PROJECTIONS:
            container = (
                "self_attn" if projection in ATTENTION_PROJECTIONS else "mlp"
            )
            names.add(
                f"model.layers.{layer_index}.{container}.{projection}"
            )
    return names


def validate_config(config: dict[str, Any]) -> None:
    for key in ("profile", "model", "quant_database", "budget", "search", "results_root"):
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    database = config["quant_database"]
    budget = config["budget"]
    search = config["search"]
    if sorted(database["bitwidths"]) != [2, 3, 4, 5, 6]:
        raise ValueError("Apples-to-apples database must provide bit-widths 2, 3, 4, 5, 6.")
    if database["group_size"] != 128:
        raise ValueError("Apples-to-apples GPTQ group size must be 128.")
    if database["expected_modules"] != 224:
        raise ValueError("Mistral apples-to-apples scope must contain 224 projection modules.")
    if database.get("expected_reconstruction_dtype") != "float16":
        raise ValueError("Mistral reconstruction database must store FP16 tensors.")
    if database.get("expected_reconstruction_shapes") != (
        EXPECTED_MISTRAL_RECONSTRUCTION_SHAPES
    ):
        raise ValueError("Mistral reconstruction shapes do not match the architecture.")
    shape_parameter_count = 32 * sum(
        shape[0] * shape[1]
        for shape in database["expected_reconstruction_shapes"].values()
    )
    if shape_parameter_count != budget["reference_quantized_parameters"]:
        raise ValueError(
            "Configured reconstruction shapes do not match searched parameters."
        )
    if budget["uniform_target_bitwidth"] != 3:
        raise ValueError("The common reference target must be uniform 3-bit GPTQ.")
    if budget["mode"] != "match_uniform_quantization_total":
        raise ValueError("The thesis configurations must use exact total-model budgeting.")
    if len(search["survivors"]) != len(search["selection_tokens"]):
        raise ValueError("Selection survivor and token schedules must have equal length.")
    if search["survivors"][-1] != 1:
        raise ValueError("The final selection stage must retain one candidate.")
    if search["group_rule"] != "size":
        raise ValueError("Paper-compatible quantization switches require size grouping.")
    if config["dtype"] != "float16" or budget["dense_dtype_bits"] != 16:
        raise ValueError("The Mistral reference accounting requires FP16 fixed parameters.")
    if budget["reference_quantized_parameters"] + budget["reference_fixed_parameters"] != budget["reference_dense_parameters"]:
        raise ValueError("Reference fixed and quantized parameter counts do not sum to dense parameters.")
    expected_dense_bits = budget["reference_dense_parameters"] * budget["dense_dtype_bits"]
    if budget["reference_dense_bits"] != expected_dense_bits:
        raise ValueError("Configured dense cost is inconsistent with the parameter count.")
    expected_weight_only = (
        budget["reference_fixed_parameters"] * budget["dense_dtype_bits"]
        + budget["reference_quantized_parameters"] * budget["uniform_target_bitwidth"]
    )
    if budget["reference_paper_weight_only_target_bits"] != expected_weight_only:
        raise ValueError("Configured weight-only target is internally inconsistent.")
    metadata_bits = (
        budget["reference_quantized_parameters"]
        // database["group_size"]
        * (budget["scale_bits"] + budget["zero_point_bits"])
    )
    expected_target = expected_weight_only + metadata_bits
    if budget["reference_metadata_inclusive_target_bits"] != expected_target:
        raise ValueError("Configured metadata-inclusive target is internally inconsistent.")
    quant_compute = expected_search_compute(config, "quant_only")
    joint_compute = expected_search_compute(config, "joint")
    if quant_compute != joint_compute:
        raise ValueError(
            "Quant-only and joint search-compute expectations differ: "
            f"quant_only={quant_compute}, joint={joint_compute}."
        )


def default_quant_db(config: dict[str, Any], root_override: str | None) -> Path:
    save_root = Path(root_override or config["quant_database"]["save_root"])
    if not save_root.is_absolute():
        save_root = REPO_ROOT / save_root
    model_basename = config["model"].rsplit("/", 1)[-1]
    bitwidth = config["quant_database"]["calibration_bitwidth"]
    return save_root / model_basename / f"{bitwidth}bit"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    buffer = bytearray(8 * 1024**2)
    view = memoryview(buffer)
    with path.open("rb", buffering=0) as handle:
        while True:
            bytes_read = handle.readinto(buffer)
            if not bytes_read:
                break
            digest.update(view[:bytes_read])
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
    return digest.hexdigest()


def validate_quant_database(
    path: Path,
    config: dict[str, Any],
    *,
    allow_unmanifested: bool,
    expected_attempt_id: str | None = None,
) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Quantization database does not exist: {path}")
    database = config["quant_database"]
    budget = config["budget"]
    module_dirs = sorted(item for item in path.iterdir() if item.is_dir())
    if len(module_dirs) != database["expected_modules"]:
        raise ValueError(
            f"Expected {database['expected_modules']} quantized modules, found {len(module_dirs)}."
        )
    actual_module_names = {item.name for item in module_dirs}
    expected_module_names = expected_mistral_module_names()
    if actual_module_names != expected_module_names:
        raise ValueError(
            "Database module names do not exactly match the Mistral seven-projection "
            f"scope: missing={sorted(expected_module_names - actual_module_names)}, "
            f"unexpected={sorted(actual_module_names - expected_module_names)}."
        )
    manifest_path = path / "quant_database_manifest.json"
    allowed_root_entry_names = set(expected_module_names)
    if manifest_path.is_file():
        allowed_root_entry_names.add(manifest_path.name)
    actual_root_entry_names = {item.name for item in path.iterdir()}
    if actual_root_entry_names != allowed_root_entry_names:
        raise ValueError(
            "Quantization database root contains entries outside the exact module "
            "and manifest inventory: "
            f"missing={sorted(allowed_root_entry_names - actual_root_entry_names)}, "
            f"unexpected={sorted(actual_root_entry_names - allowed_root_entry_names)}."
        )
    expected_levels = sorted(database["bitwidths"])
    failures = {}
    unreadable_files = {}
    observed_metadata = {}
    projection_counts: Counter[str] = Counter()
    expected_level_filenames = {f"{level}.pth" for level in expected_levels}
    unexpected_module_contents = {}
    for module_dir in module_dirs:
        actual_entry_names = {item.name for item in module_dir.iterdir()}
        if actual_entry_names != expected_level_filenames:
            unexpected_module_contents[module_dir.name] = {
                "missing": sorted(expected_level_filenames - actual_entry_names),
                "unexpected": sorted(actual_entry_names - expected_level_filenames),
            }
        level_paths = sorted(
            (item for item in module_dir.glob("*.pth") if item.stem.isdigit()),
            key=lambda item: int(item.stem),
        )
        levels = [int(item.stem) for item in level_paths]
        if levels != expected_levels:
            failures[module_dir.name] = levels
        observed_metadata[module_dir.name] = {}
        projection = module_dir.name.rsplit(".", 1)[-1]
        expected_shape = database["expected_reconstruction_shapes"][projection]
        expected_dtype = database["expected_reconstruction_dtype"]
        for level_path in level_paths:
            if level_path.stat().st_size <= 0:
                unreadable_files[str(level_path)] = "empty file"
                continue
            try:
                tensor = torch.load(
                    level_path,
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
            except Exception as error:  # noqa: BLE001 - report corrupt artifacts
                unreadable_files[str(level_path)] = repr(error)
                continue
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                unreadable_files[str(level_path)] = (
                    f"expected a non-empty Tensor, got {type(tensor).__name__}"
                )
                del tensor
                continue
            finite = bool(torch.isfinite(tensor).all())
            metadata = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "numel": int(tensor.numel()),
                "contiguous": bool(tensor.is_contiguous()),
                "file_size_bytes": level_path.stat().st_size,
                "file_sha256": sha256_file(level_path),
            }
            observed_metadata[module_dir.name][level_path.stem] = metadata
            if metadata["shape"] != expected_shape:
                unreadable_files[str(level_path)] = (
                    f"shape {metadata['shape']} != expected {expected_shape}"
                )
            elif metadata["dtype"] != expected_dtype:
                unreadable_files[str(level_path)] = (
                    f"dtype {metadata['dtype']} != expected {expected_dtype}"
                )
            elif not metadata["contiguous"]:
                unreadable_files[str(level_path)] = "tensor is not contiguous"
            elif tensor.device.type != "cpu":
                unreadable_files[str(level_path)] = (
                    f"tensor loaded on unexpected device {tensor.device}"
                )
            elif not finite:
                unreadable_files[str(level_path)] = "tensor contains non-finite values"
            del tensor
        projection_counts[projection] += 1
    if failures:
        raise ValueError(f"Database levels do not match 2--6 for every module: {failures}")
    if unexpected_module_contents:
        raise ValueError(
            "Each quantized module directory must contain exactly the five "
            "expected reconstruction files: "
            f"{unexpected_module_contents}"
        )
    if unreadable_files:
        raise ValueError(
            "Database contains empty, corrupt, or non-tensor reconstruction files: "
            f"{unreadable_files}"
        )
    expected_projection_counts = Counter({projection: 32 for projection in PROJECTIONS})
    if projection_counts != expected_projection_counts:
        raise ValueError(
            "Database is not the full seven-projection Mistral scope: "
            f"actual={dict(projection_counts)}."
        )

    if not manifest_path.is_file():
        if allow_unmanifested:
            return
        raise ValueError(
            f"Missing {manifest_path}. Regenerate with the apples launcher or pass "
            "--allow-unmanifested-db only after independently verifying provenance."
        )
    manifest = read_json(manifest_path)
    if not isinstance(manifest.get("attempt_id"), str) or not manifest["attempt_id"]:
        raise ValueError("Quantization database manifest is missing its attempt ID.")
    if (
        expected_attempt_id is not None
        and manifest["attempt_id"] != expected_attempt_id
    ):
        raise ValueError(
            "Quantization database attempt ID does not match its launcher: "
            f"expected={expected_attempt_id}, actual={manifest['attempt_id']}."
        )
    expected_values = {
        "status": "complete",
        "model_name": config["model"],
        "tokenizer_name": config["model"],
        "tokenizer_is_fast": False,
        "attention_implementation": database["attention_implementation"],
        "bitwidth_options": expected_levels,
        "group_size": database["group_size"],
        "perchannel": database["perchannel"],
        "symmetric": database["symmetric"],
        "activation_order": database["activation_order"],
        "calibration_data": database["calibration_data"],
        "calibration_tokens": database["calibration_tokens"],
        "calibration_sequence_length": database["sequence_length"],
        "calibration_token_count_loaded": database["calibration_tokens"],
        "calibration_logical_shard_count": database["torchrun_processes"],
        "configured_torchrun_processes": database["torchrun_processes"],
        "module_count": database["expected_modules"],
        "total_parameters_dense": budget["reference_dense_parameters"],
        "quantized_weight_parameters": budget["reference_quantized_parameters"],
        "fixed_parameters": budget["reference_fixed_parameters"],
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_values.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Quantization database manifest mismatch: {mismatches}")
    size_inventory = manifest.get("level_file_sizes_bytes")
    if not isinstance(size_inventory, dict) or set(size_inventory) != {
        item.name for item in module_dirs
    }:
        raise ValueError("Manifest reconstruction-file inventory is missing or incomplete.")
    inventory_mismatches = {}
    for module_dir in module_dirs:
        module_inventory = size_inventory.get(module_dir.name)
        if not isinstance(module_inventory, dict):
            inventory_mismatches[module_dir.name] = "missing module inventory"
            continue
        actual_sizes = {
            str(level): (module_dir / f"{level}.pth").stat().st_size
            for level in expected_levels
        }
        if module_inventory != actual_sizes:
            inventory_mismatches[module_dir.name] = {
                "manifest": module_inventory,
                "actual": actual_sizes,
            }
    if inventory_mismatches:
        raise ValueError(
            "Database reconstruction-file sizes do not match the manifest: "
            f"{inventory_mismatches}"
        )
    if manifest.get("reconstruction_metadata") != observed_metadata:
        raise ValueError(
            "Database reconstruction shape/dtype/hash metadata does not match files."
        )
    observed_payload_bytes = sum(
        metadata["file_size_bytes"]
        for levels in observed_metadata.values()
        for metadata in levels.values()
    )
    if manifest.get("database_payload_bytes") != observed_payload_bytes:
        raise ValueError(
            "Database payload-byte total does not match reconstruction files."
        )
    configured_processes = manifest.get("configured_torchrun_processes")
    effective_processes = manifest.get("distributed_world_size")
    if not isinstance(configured_processes, int) or configured_processes < 1:
        raise ValueError(
            "Manifest configured_torchrun_processes must be a positive integer."
        )
    if not isinstance(effective_processes, int) or effective_processes < 1:
        raise ValueError("Manifest distributed_world_size must be a positive integer.")
    if manifest.get("torchrun_process_override") != (
        configured_processes != effective_processes
    ):
        raise ValueError("Manifest torchrun process-override provenance is inconsistent.")
    memory_mode = manifest.get("database_memory_mode")
    if memory_mode not in {"in_memory_activations", "disk_activation_cache"}:
        raise ValueError(f"Unsupported or missing database memory mode: {memory_mode!r}.")
    if memory_mode == "disk_activation_cache":
        if effective_processes != 1:
            raise ValueError(
                "Disk-backed database generation must have effective world size 1."
            )
        if manifest.get("model_residency") != "gpu_direct":
            raise ValueError(
                "Disk-backed database generation must verify direct GPU model residency."
            )
        if manifest.get("gpu_direct_residency_verified") is not True:
            raise ValueError(
                "Disk-backed manifest does not confirm parameter/buffer GPU residency."
            )
        cache_summary = manifest.get("activation_cache_summary")
        if not isinstance(cache_summary, dict):
            raise ValueError("Disk-backed manifest is missing activation-cache summary.")
        if cache_summary.get("sample_count") != manifest.get(
            "calibration_sequence_count_per_rank"
        ):
            raise ValueError(
                "Activation-cache sample count does not match calibration sequences."
            )
        if not isinstance(cache_summary.get("current_size_bytes"), int) or (
            cache_summary["current_size_bytes"] <= 0
        ):
            raise ValueError("Activation-cache size provenance is missing or invalid.")
        if expected_attempt_id is not None:
            cache_path = Path(str(cache_summary.get("path", "")))
            cache_manifest_path = cache_path / "activation_cache_manifest.json"
            if not cache_manifest_path.is_file():
                raise ValueError(
                    "Prepare-db postflight cannot find the activation-cache manifest: "
                    f"{cache_manifest_path}."
                )
            cache_manifest = read_json(cache_manifest_path)
            expected_cache_values = {
                "attempt_id": expected_attempt_id,
                "status": "complete",
                "sample_count": cache_summary["sample_count"],
                "current_size_bytes": cache_summary["current_size_bytes"],
                "completed_blocks": 32,
                "total_blocks": 32,
            }
            cache_mismatches = {
                key: {"expected": value, "actual": cache_manifest.get(key)}
                for key, value in expected_cache_values.items()
                if cache_manifest.get(key) != value
            }
            if cache_mismatches:
                raise ValueError(
                    "Activation-cache completion manifest mismatch: "
                    f"{cache_mismatches}"
                )
    loaded_tokens = manifest.get("calibration_token_count_loaded")
    used_tokens = manifest.get("calibration_token_count_used")
    if loaded_tokens != database["calibration_tokens"]:
        raise ValueError(
            "Manifest loaded calibration-token count does not match the request: "
            f"expected={database['calibration_tokens']}, actual={loaded_tokens}."
        )
    if not isinstance(used_tokens, int) or not 0 < used_tokens <= loaded_tokens:
        raise ValueError(
            "Manifest used calibration-token count must be positive and no larger "
            "than the loaded count."
        )
    if manifest.get("calibration_token_digest_algorithm") != (
        "sha256-v1-dtype-shape-boundaries"
    ):
        raise ValueError("Manifest calibration-token digest algorithm is missing.")
    for digest_key in (
        "calibration_token_digest_loaded",
        "calibration_token_digest_used",
    ):
        digest_value = manifest.get(digest_key)
        if not isinstance(digest_value, str) or re.fullmatch(
            r"[0-9a-f]{64}", digest_value
        ) is None:
            raise ValueError(f"Manifest {digest_key} is missing or invalid.")
    logical_token_counts = manifest.get("calibration_token_counts_by_logical_shard")
    if (
        not isinstance(logical_token_counts, list)
        or len(logical_token_counts) != configured_processes
        or any(not isinstance(value, int) or value <= 0 for value in logical_token_counts)
        or sum(logical_token_counts) != used_tokens
    ):
        raise ValueError(
            "Manifest logical-shard token counts are missing or inconsistent."
        )


def common_eval_command(config: dict[str, Any], seed: int, python_bin: str) -> list[str]:
    search = config["search"]
    command = [
        python_bin,
        "eval_ppl.py",
        "--model_name_or_path",
        config["model"],
        "--eval_datasets",
        *search["eval_datasets"],
        "--eval_tokens",
        str(search["eval_tokens"]),
        "--sequence_length",
        str(search["eval_sequence_length"]),
        "--dtype",
        config["dtype"],
        "--attn_implementation",
        config["attention_implementation"],
        "--seed",
        str(seed),
    ]
    if config.get("use_fast_tokenizer", False):
        command.append("--use_fast_tokenizer")
    return command


def common_search_args(
    config: dict[str, Any],
    seed: int,
    quant_db: Path,
    output_dir: Path,
) -> list[str]:
    search = config["search"]
    database = config["quant_database"]
    budget = config["budget"]
    args = [
        "--model_name_or_path",
        config["model"],
        "--quant_weights_path",
        str(quant_db),
        "--target_bitwidth",
        str(budget["uniform_target_bitwidth"]),
        "--calibration_data",
        search["calibration_data"],
        "--calibration_tokens",
        str(search["calibration_tokens"]),
        "--calibration_sequence_length",
        str(search["sequence_length"]),
        "--eval_datasets",
        *search["eval_datasets"],
        "--eval_tokens",
        str(search["eval_tokens"]),
        "--eval_sequence_length",
        str(search["eval_sequence_length"]),
        "--eval_every",
        str(search["eval_every"]),
        "--generations",
        str(search["generations"]),
        "--offspring",
        str(search["offspring"]),
        "--survivors_per_selection",
        *[str(value) for value in search["survivors"]],
        "--tokens_per_selection",
        *[str(value) for value in search["selection_tokens"]],
        "--initial_tokens",
        str(search["initial_tokens"]),
        "--fitness_fn",
        search["fitness"],
        "--group_rule",
        search["group_rule"],
        "--step_size",
        str(search["step_size"]),
        "--compression_budget_mode",
        budget["mode"],
        "--quantization_group_size",
        str(database["group_size"]),
        "--budget_scale_bits",
        str(budget["scale_bits"]),
        "--budget_zero_point_bits",
        str(budget["zero_point_bits"]),
        "--budget_dense_dtype_bits",
        str(budget["dense_dtype_bits"]),
        "--expected_dense_model_bits",
        str(budget["reference_dense_bits"]),
        "--expected_target_cost_bits",
        str(budget["reference_metadata_inclusive_target_bits"]),
        "--expected_bitwidths",
        *[str(value) for value in database["bitwidths"]],
        "--expected_quantized_modules",
        str(database["expected_modules"]),
        "--dtype",
        config["dtype"],
        "--attn_implementation",
        config["attention_implementation"],
        "--seed",
        str(seed),
        "--output_dir",
        str(output_dir),
    ]
    if budget["include_quantization_metadata"]:
        args.append("--budget_include_quantization_metadata")
    if config.get("use_fast_tokenizer", False):
        args.append("--use_fast_tokenizer")
    return args


def build_command(
    method: str,
    config: dict[str, Any],
    seed: int,
    quant_db: Path,
    output_dir: Path,
    python_bin: str,
    torchrun_bin: str,
    quant_db_root: str | None,
    torchrun_processes: int | None = None,
    database_memory_mode: str = "legacy_cpu_offload",
    activation_cache_dir: Path | None = None,
    attempt_id: str | None = None,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    database = config["quant_database"]
    search = config["search"]
    budget = config["budget"]
    if method == "prepare_db":
        effective_torchrun_processes = (
            database["torchrun_processes"]
            if torchrun_processes is None
            else torchrun_processes
        )
        if effective_torchrun_processes < 1:
            raise ValueError("Database generation requires at least one torchrun process.")
        if database_memory_mode not in {
            "legacy_cpu_offload",
            "disk_activation_cache",
        }:
            raise ValueError(f"Unknown database memory mode: {database_memory_mode}")
        if database_memory_mode == "disk_activation_cache":
            if effective_torchrun_processes != 1:
                raise ValueError(
                    "Disk-backed activation caching requires exactly one effective "
                    "torchrun process."
                )
            if activation_cache_dir is None:
                raise ValueError(
                    "Disk-backed activation caching requires activation_cache_dir."
                )
        model_basename = config["model"].rsplit("/", 1)[-1]
        expected_relative_target = (
            Path(model_basename) / f"{database['calibration_bitwidth']}bit"
        )
        save_root = quant_db.parent.parent
        if save_root / expected_relative_target != quant_db:
            raise ValueError(
                "Resolved prepare_db target does not match the launcher's expected "
                f"layout: target={quant_db}, expected_suffix={expected_relative_target}."
            )
        command = [
            torchrun_bin,
            "--nnodes=1",
            f"--nproc-per-node={effective_torchrun_processes}",
            "quant.py",
            "--configured_torchrun_processes",
            str(database["torchrun_processes"]),
            "--model_name_or_path",
            config["model"],
            "--quantizable_modules",
            database["quantizable_modules"],
            "--expected_module_count",
            str(database["expected_modules"]),
            "--expected_reconstruction_dtype",
            database["expected_reconstruction_dtype"],
            "--pre_block_modules",
            "model.embed_tokens",
            "--block_modules",
            "model.layers",
            "--post_block_modules",
            "model.norm",
            "lm_head",
            "--calibration_data",
            database["calibration_data"],
            "--calibration_tokens",
            str(database["calibration_tokens"]),
            "--calibration_sequence_length",
            str(database["sequence_length"]),
            "--bitwidth_options",
            *[str(value) for value in database["bitwidths"]],
            "--calibration_bitwidth",
            str(database["calibration_bitwidth"]),
            "--group_size",
            str(database["group_size"]),
            "--rel_damp",
            str(database["relative_dampening"]),
            "--block_size",
            str(database["block_size"]),
            "--seed",
            str(seed),
            "--low_cpu_mem_usage",
            "--cpu_offload_activations",
            "--drop_saved_file_cache",
            "--verbose",
            "--dtype",
            config["dtype"],
            "--attn_implementation",
            database["attention_implementation"],
            "--save_dir",
            str(save_root),
            "--perchannel",
        ]
        if attempt_id is not None:
            command.extend(["--attempt_id", attempt_id])
        if database_memory_mode == "legacy_cpu_offload":
            command.append("--cpu_offload_modules")
        else:
            command.extend(
                [
                    "--load_model_to_gpu",
                    "--activation_cache_dir",
                    str(activation_cache_dir),
                ]
            )
        return command
    if method == "dense":
        return common_eval_command(config, seed, python_bin)
    if method == "uniform3":
        return common_eval_command(config, seed, python_bin) + [
            "--quant_weights_path",
            str(quant_db),
            "--quant_default_level",
            str(budget["uniform_target_bitwidth"]),
        ]
    common = common_search_args(config, seed, quant_db, output_dir)
    if method == "quant_only":
        command = [python_bin, "evo_quant_search.py", *common]
        command += ["--initially_generated", str(search["quant_initial_candidates"])]
        if search["skip_quant_uniform_initial_evaluation"]:
            command.append("--skip_initial_uniform_evaluation")
        if resume_checkpoint is not None:
            command.extend(
                ["--resume_checkpoint", str(resume_checkpoint)]
            )
        return command
    if method == "joint":
        command = [python_bin, "evo_joint_search.py", *common]
        command += [
            "--drop_sparsity",
            str(budget["joint_drop_sparsity"]),
            "--max_drop_mutations",
            str(search["max_drop_mutations"]),
            "--joint_mutation_mode",
            search["joint_mutation_mode"],
            "--initially_generated",
            str(search["joint_initial_candidates"]),
            "--population_size",
            str(search["population_size"]),
            "--crossover_probability",
            str(search["crossover_probability"]),
        ]
        if budget["drop_entire_block"]:
            command.append("--drop_entire_block")
        if search.get("skip_joint_single_initial_evaluation", False):
            command.append("--skip_initial_single_candidate_evaluation")
        if resume_checkpoint is not None:
            command.extend(
                ["--resume_checkpoint", str(resume_checkpoint)]
            )
        return command
    raise ValueError(f"Unsupported method: {method}")


def parse_perplexities(log_text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    pattern = re.compile(r"^(wikitext2|c4|fineweb_edu):\s+([0-9]+(?:\.[0-9]+)?)\s*$")
    for line in log_text.splitlines():
        match = pattern.match(line.strip())
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def eval_summary(
    method: str,
    config: dict[str, Any],
    seed: int,
    runtime_seconds: float,
    metrics: dict[str, float],
    output_dir: Path,
    quant_db: Path,
) -> dict[str, Any]:
    budget = config["budget"]
    is_dense = method == "dense"
    cost_bits = (
        budget["reference_dense_bits"]
        if is_dense
        else budget["reference_metadata_inclusive_target_bits"]
    )
    ratio = 1.0 if is_dense else budget["reference_metadata_inclusive_compression_ratio"]
    bitwidth_by_module = (
        {
            path.name: budget["uniform_target_bitwidth"]
            for path in sorted(quant_db.iterdir())
            if path.is_dir()
        }
        if method == "uniform3"
        else {}
    )
    candidate = {
        "candidate_type": "dense" if is_dense else "uniform_quantization",
        "dropped_modules": [],
        "attention_mask": [0] * 32,
        "mlp_mask": [0] * 32,
        "bitwidth_by_module": bitwidth_by_module,
        "candidate_vector_raw": None if is_dense else "uniform_3_bit",
    }
    candidate_path = output_dir / "final_candidate.json"
    write_json(candidate_path, candidate)
    return {
        "schema_version": 3,
        "run_name": output_dir.name,
        "timestamp_end": utc_now(),
        "git_commit": git_commit(),
        "model_name": config["model"],
        "dataset_calibration": None,
        "dataset_eval": config["search"]["eval_datasets"],
        "search_type": "dense" if is_dense else "uniform_quantization",
        "search_config": {"seed": seed, "candidate_evaluations_search_total": 0},
        "compression_config": {
            "target_cost_bits": cost_bits,
            "compression_budget_mode": budget["mode"],
            "target_average_bitwidth": None if is_dense else 3,
            "include_quantization_metadata": not is_dense,
        },
        "final_metrics": {
            "wikitext2_ppl": metrics.get("wikitext2"),
            "c4_ppl": metrics.get("c4"),
            "fineweb_ppl": metrics.get("fineweb_edu"),
            "best_search_fitness": None,
            "final_calibration_kl": None,
            "runtime_seconds": runtime_seconds,
            "compression_target_bits": cost_bits,
            "compression_realized_bits": cost_bits,
            "compression_difference_bits": 0,
            "estimated_compression_ratio": ratio,
            "estimated_weight_memory_mb": cost_bits / 8 / 1024**2,
        },
        "model_size_statistics": {
            "compression_cost_bits": cost_bits,
            "estimated_weight_memory_mb": cost_bits / 8 / 1024**2,
            "dense_weight_memory_mb": budget["reference_dense_bits"] / 8 / 1024**2,
            "estimated_compression_ratio": ratio,
        },
        "quantization_statistics": {
            "quantized_module_count": len(bitwidth_by_module),
            "bitwidth_by_module": bitwidth_by_module,
        },
        "artifacts": {
            "stdout_log_path": str(output_dir / "run.log"),
            "candidate_path": str(candidate_path),
            "resolved_config_path": str(output_dir / "resolved_config.json"),
        },
    }


def validate_search_run_summary(
    summary_path: Path,
    method: str,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Require the structured artifacts that make a search run scientifically usable."""

    expected_search_type = {
        "quant_only": "quant_only",
        "joint": "joint_depth_quant",
    }.get(method)
    if expected_search_type is None:
        raise ValueError(f"Search-summary validation is unsupported for {method!r}.")
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Completed {method} child did not produce required run summary: "
            f"{summary_path}"
        )

    summary = read_json(summary_path)
    expected_target_bits = config["budget"][
        "reference_metadata_inclusive_target_bits"
    ]
    top_level_expectations = {
        "search_type": expected_search_type,
        "model_name": config["model"],
        "dataset_eval": config["search"]["eval_datasets"],
    }
    top_level_mismatches = {
        key: {"expected": expected, "actual": summary.get(key)}
        for key, expected in top_level_expectations.items()
        if summary.get(key) != expected
    }
    if top_level_mismatches:
        raise ValueError(
            f"Completed {method} summary identity mismatch: {top_level_mismatches}"
        )

    search_config = summary.get("search_config")
    if not isinstance(search_config, dict):
        raise ValueError(f"Completed {method} summary is missing search_config.")
    expected_compute = expected_search_compute(config, method)
    search_expectations = {
        "seed": seed,
        "generations": config["search"]["generations"],
        "offspring": config["search"]["offspring"],
        "fitness_fn": config["search"]["fitness"],
        "candidate_evaluations_search_total": expected_compute[
            "candidate_evaluations"
        ],
        "evaluation_tokens_search_total": expected_compute["candidate_tokens"],
    }
    search_mismatches = {
        key: {"expected": expected, "actual": search_config.get(key)}
        for key, expected in search_expectations.items()
        if search_config.get(key) != expected
    }
    if search_mismatches:
        raise ValueError(
            f"Completed {method} summary search-compute mismatch: "
            f"{search_mismatches}"
        )

    compression_config = summary.get("compression_config")
    if not isinstance(compression_config, dict):
        raise ValueError(f"Completed {method} summary is missing compression_config.")
    compression_expectations = {
        "compression_budget_mode": config["budget"]["mode"],
        "target_cost_bits": expected_target_bits,
        "database_module_count": config["quant_database"]["expected_modules"],
    }
    compression_mismatches = {
        key: {"expected": expected, "actual": compression_config.get(key)}
        for key, expected in compression_expectations.items()
        if compression_config.get(key) != expected
    }
    if compression_mismatches:
        raise ValueError(
            f"Completed {method} summary compression mismatch: "
            f"{compression_mismatches}"
        )

    final_metrics = summary.get("final_metrics")
    if not isinstance(final_metrics, dict):
        raise ValueError(f"Completed {method} summary is missing final_metrics.")
    metric_expectations = {
        "compression_target_bits": expected_target_bits,
        "compression_realized_bits": expected_target_bits,
        "compression_difference_bits": 0,
        "exact_budget_valid": True,
    }
    metric_mismatches = {
        key: {"expected": expected, "actual": final_metrics.get(key)}
        for key, expected in metric_expectations.items()
        if final_metrics.get(key) != expected
    }
    if metric_mismatches:
        raise ValueError(
            f"Completed {method} summary exact-budget mismatch: {metric_mismatches}"
        )
    required_metric_names = {
        "wikitext2": "wikitext2_ppl",
        "c4": "c4_ppl",
        "fineweb_edu": "fineweb_ppl",
    }
    required_metrics = [
        required_metric_names[dataset]
        for dataset in config["search"]["eval_datasets"]
    ]
    if config["search"]["fitness"] == "kl":
        required_metrics.append("final_calibration_kl")
    invalid_metrics = {
        name: final_metrics.get(name)
        for name in required_metrics
        if isinstance(final_metrics.get(name), bool)
        or not isinstance(final_metrics.get(name), (int, float))
        or not math.isfinite(final_metrics[name])
        or final_metrics[name] < 0
    }
    if invalid_metrics:
        raise ValueError(
            f"Completed {method} summary is missing required finite metrics: "
            f"{invalid_metrics}"
        )

    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"Completed {method} summary is missing artifacts.")
    missing_artifacts = {}
    for key in ("candidate_path", "generation_log_path", "config_path"):
        value = artifacts.get(key)
        if not isinstance(value, str) or not Path(value).is_file():
            missing_artifacts[key] = value
    if missing_artifacts:
        raise ValueError(
            f"Completed {method} summary references missing artifacts: "
            f"{missing_artifacts}"
        )
    return summary


def _linux_process_tree_rss_snapshot(root_pid: int) -> dict[str, Any]:
    """Sample aggregate RSS for a Linux process and all current descendants."""

    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return {
            "supported": False,
            "root_pid": root_pid,
            "process_count": None,
            "rss_bytes": None,
            "pids": [],
        }

    processes: dict[int, tuple[int, int | None]] = {}
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        entries = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            stat_text = (entry / "stat").read_text(encoding="utf-8")
            closing_parenthesis = stat_text.rfind(")")
            if closing_parenthesis < 0:
                continue
            stat_fields = stat_text[closing_parenthesis + 1 :].split()
            parent_pid = int(stat_fields[1])
            rss_bytes = None
            for line in (entry / "status").read_text(
                encoding="utf-8"
            ).splitlines():
                if line.startswith("VmRSS:"):
                    rss_bytes = int(line.split()[1]) * 1024
                    break
            processes[pid] = (parent_pid, rss_bytes)
        except (IndexError, OSError, ValueError):
            continue

    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent_pid, _) in processes.items():
            if parent_pid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    live_pids = sorted(pid for pid in descendants if pid in processes)
    rss_values = [processes[pid][1] for pid in live_pids]
    known_rss_values = [value for value in rss_values if value is not None]
    return {
        "supported": True,
        "root_pid": root_pid,
        "process_count": len(live_pids),
        "rss_bytes": (
            sum(known_rss_values) if len(known_rss_values) == len(live_pids) else None
        ),
        "pids": live_pids,
    }


def _process_group_exists(process_group_id: int) -> bool:
    if os.name != "posix":
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(
    process: subprocess.Popen[str],
    process_group_id: int,
    *,
    grace_seconds: float = 5.0,
) -> None:
    """Terminate the whole launched command tree and reap its direct child."""

    if os.name == "posix":
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace_seconds
        while _process_group_exists(process_group_id) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.05)
        if _process_group_exists(process_group_id):
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
    elif process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
    try:
        process.wait(timeout=max(1.0, grace_seconds))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_and_tee(
    command: list[str],
    log_path: Path,
    resource_log_path: Path,
    *,
    attempt_id: str | None = None,
) -> tuple[int, dict[str, Any]]:
    child_environment = dict(os.environ)
    child_environment["PYTHONUNBUFFERED"] = "1"
    process: subprocess.Popen[str] | None = None
    process_group_id: int | None = None
    sampler: threading.Thread | None = None
    stop_sampling = threading.Event()
    sampler_errors: list[BaseException] = []
    primary_error: BaseException | None = None
    exit_code: int | None = None
    sampling_summary: dict[str, Any] = {}
    baseline: dict[str, Any] = {}
    with log_path.open("x", encoding="utf-8") as handle:
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=child_environment,
                start_new_session=(os.name == "posix"),
            )
            process_group_id = process.pid
            baseline = cgroup_memory_snapshot()
            initial_process_tree = _linux_process_tree_rss_snapshot(process.pid)
            sampling_summary = {
                "attempt_id": attempt_id,
                "child_pid": process.pid,
                "process_group_id": process_group_id,
                "sampling_interval_seconds": 5,
                "baseline": baseline,
                "max_current_bytes_observed": baseline.get("current_bytes"),
                "max_peak_bytes_observed": baseline.get("peak_bytes"),
                "max_process_tree_rss_bytes_observed": initial_process_tree.get(
                    "rss_bytes"
                ),
                "max_process_count_observed": initial_process_tree.get(
                    "process_count"
                ),
            }

            def sample_resources() -> None:
                try:
                    with resource_log_path.open(
                        "x", encoding="utf-8"
                    ) as resource_handle:
                        while True:
                            snapshot = cgroup_memory_snapshot()
                            process_tree = _linux_process_tree_rss_snapshot(
                                process.pid
                            )
                            record = {
                                "attempt_id": attempt_id,
                                "timestamp": utc_now(),
                                "monotonic_seconds": time.monotonic(),
                                "child_pid": process.pid,
                                "process_group_id": process_group_id,
                                "child_poll": process.poll(),
                                "cgroup_memory": snapshot,
                                "process_tree": process_tree,
                            }
                            resource_handle.write(
                                json.dumps(record, sort_keys=True, allow_nan=False)
                                + "\n"
                            )
                            resource_handle.flush()
                            os.fsync(resource_handle.fileno())
                            for source_key, destination_key in (
                                ("current_bytes", "max_current_bytes_observed"),
                                ("peak_bytes", "max_peak_bytes_observed"),
                            ):
                                value = snapshot.get(source_key)
                                previous = sampling_summary.get(destination_key)
                                if value is not None and (
                                    previous is None or value > previous
                                ):
                                    sampling_summary[destination_key] = value
                            rss_bytes = process_tree.get("rss_bytes")
                            if rss_bytes is not None and (
                                sampling_summary[
                                    "max_process_tree_rss_bytes_observed"
                                ]
                                is None
                                or rss_bytes
                                > sampling_summary[
                                    "max_process_tree_rss_bytes_observed"
                                ]
                            ):
                                sampling_summary[
                                    "max_process_tree_rss_bytes_observed"
                                ] = rss_bytes
                            process_count = process_tree.get("process_count")
                            if process_count is not None and (
                                sampling_summary["max_process_count_observed"] is None
                                or process_count
                                > sampling_summary["max_process_count_observed"]
                            ):
                                sampling_summary[
                                    "max_process_count_observed"
                                ] = process_count
                            if stop_sampling.wait(5):
                                break
                except BaseException as error:  # noqa: BLE001 - propagate from thread
                    sampler_errors.append(error)
                    stop_sampling.set()
                    if process_group_id is not None:
                        _terminate_process_group(process, process_group_id)

            sampler = threading.Thread(
                target=sample_resources,
                name="cgroup-memory-sampler",
                daemon=True,
            )
            sampler.start()
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                handle.write(line)
                handle.flush()
            exit_code = process.wait()
        except BaseException as error:  # noqa: BLE001 - cleanup before re-raising
            primary_error = error
        finally:
            stop_sampling.set()
            if process is not None and process_group_id is not None and (
                primary_error is not None
                or sampler_errors
                or process.poll() is None
            ):
                _terminate_process_group(process, process_group_id)
            if sampler is not None:
                sampler.join(timeout=15)
                if sampler.is_alive():
                    sampler_errors.append(
                        RuntimeError("Resource sampler did not stop within 15 seconds.")
                    )
                    if process is not None and process_group_id is not None:
                        _terminate_process_group(process, process_group_id)
            if process is not None and sampler_errors and process_group_id is not None:
                _terminate_process_group(process, process_group_id)
            if process is not None and process.stdout is not None:
                process.stdout.close()
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException as error:  # noqa: BLE001 - preserve logging failure
                if primary_error is None:
                    primary_error = error

    if primary_error is not None:
        raise primary_error
    if sampler_errors:
        raise RuntimeError(
            "Resource sampler failed; the launched process group was terminated: "
            f"{sampler_errors!r}"
        ) from sampler_errors[0]
    if process is None or exit_code is None:
        raise RuntimeError("Launcher process ended without a child exit code.")

    final = cgroup_memory_snapshot()
    final_process_tree = _linux_process_tree_rss_snapshot(process.pid)
    sampling_summary["final"] = final
    sampling_summary["final_process_tree"] = final_process_tree
    baseline_events = baseline.get("events", {})
    final_events = final.get("events", {})
    sampling_summary["event_deltas"] = {
        key: final_events.get(key, 0) - baseline_events.get(key, 0)
        for key in sorted(set(baseline_events) | set(final_events))
    }
    return exit_code, sampling_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "method",
        choices=["prepare_db", "dense", "uniform3", "quant_only", "joint"],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quant-db", type=Path, default=None)
    parser.add_argument("--quant-db-root", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--torchrun", default="torchrun")
    parser.add_argument(
        "--torchrun-processes",
        type=int,
        default=None,
        help=(
            "Explicit prepare_db process-count override. The configured value remains "
            "unchanged and both configured/effective values are logged as provenance."
        ),
    )
    parser.add_argument(
        "--database-memory-mode",
        choices=["legacy_cpu_offload", "disk_activation_cache"],
        default="legacy_cpu_offload",
        help=(
            "Execution-only memory strategy for prepare_db. The scientific GPTQ "
            "configuration is unchanged and the selected strategy is recorded."
        ),
    )
    parser.add_argument(
        "--activation-cache-dir",
        type=Path,
        default=None,
        help="New, non-existing scratch path required by disk_activation_cache mode.",
    )
    parser.add_argument("--allow-unmanifested-db", action="store_true")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint from an interrupted quant_only or joint run. "
            "The resumed search is written into a fresh run directory."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.quant_db is not None and args.quant_db_root is not None:
        parser.error("Use only one of --quant-db and --quant-db-root.")

    if args.resume_checkpoint is not None:
        if args.method not in {"quant_only", "joint"}:
            parser.error(
                "--resume-checkpoint is valid only for quant_only or joint."
            )
        if not args.resume_checkpoint.is_file():
            parser.error(
                "--resume-checkpoint does not exist or is not a file: "
                f"{args.resume_checkpoint}"
            )
    if args.method == "prepare_db" and args.quant_db is not None:
        parser.error(
            "prepare_db accepts --quant-db-root, not --quant-db; the final target "
            "is derived exactly as <root>/<model>/<calibration-bitwidth>bit."
        )
    if args.torchrun_processes is not None:
        if args.method != "prepare_db":
            parser.error("--torchrun-processes is valid only for prepare_db.")
        if args.torchrun_processes < 1:
            parser.error("--torchrun-processes must be at least 1.")
    if args.method != "prepare_db":
        if args.database_memory_mode != "legacy_cpu_offload":
            parser.error("--database-memory-mode is valid only for prepare_db.")
        if args.activation_cache_dir is not None:
            parser.error("--activation-cache-dir is valid only for prepare_db.")
    if args.database_memory_mode == "disk_activation_cache":
        if args.activation_cache_dir is None:
            parser.error(
                "--database-memory-mode disk_activation_cache requires "
                "--activation-cache-dir."
            )
        if args.torchrun_processes != 1:
            parser.error(
                "disk_activation_cache requires --torchrun-processes 1."
            )
    elif args.activation_cache_dir is not None:
        parser.error(
            "--activation-cache-dir requires --database-memory-mode "
            "disk_activation_cache."
        )

    config_path = args.config.resolve()
    config = read_json(config_path)
    validate_config(config)
    quant_db = (
        args.quant_db.resolve()
        if args.quant_db is not None
        else default_quant_db(config, args.quant_db_root)
    )
    activation_cache_dir = (
        args.activation_cache_dir.resolve()
        if args.activation_cache_dir is not None
        else None
    )
    resume_checkpoint = (
        args.resume_checkpoint.resolve()
        if args.resume_checkpoint is not None
        else None
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = args.run_id or f"{args.method}_seed{args.seed}_{timestamp}"
    if (
        Path(run_id).name != run_id
        or run_id in {"", ".", ".."}
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None
    ):
        parser.error(
            "--run-id must be a single safe basename containing only letters, "
            "digits, '.', '_', and '-'."
        )
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        results_root = Path(config["results_root"])
        if not results_root.is_absolute():
            results_root = REPO_ROOT / results_root
        output_dir = results_root / config["profile"] / args.method / run_id

    attempt_id = uuid.uuid4().hex
    command = build_command(
        args.method,
        config,
        args.seed,
        quant_db,
        output_dir,
        args.python,
        args.torchrun,
        args.quant_db_root,
        args.torchrun_processes,
        args.database_memory_mode,
        activation_cache_dir,
        attempt_id,
        resume_checkpoint,
    )
    if args.dry_run:
        print(f"profile={config['profile']}")
        print(f"method={args.method}")
        print(f"seed={args.seed}")
        print(f"quant_db={quant_db}")
        print(f"output_dir={output_dir}")
        if args.method == "prepare_db":
            configured_processes = config["quant_database"]["torchrun_processes"]
            effective_processes = (
                configured_processes
                if args.torchrun_processes is None
                else args.torchrun_processes
            )
            print(f"configured_torchrun_processes={configured_processes}")
            print(f"effective_torchrun_processes={effective_processes}")
            print(
                "torchrun_process_override="
                f"{str(effective_processes != configured_processes).lower()}"
            )
            print(f"database_memory_mode={args.database_memory_mode}")
            print(f"activation_cache_dir={activation_cache_dir}")
            print(f"attempt_id={attempt_id}")
        print(
            "expected_search_compute="
            f"{json.dumps(expected_search_compute(config, args.method), sort_keys=True)}"
        )
        print(f"command={shlex.join(command)}")
        return 0

    if args.method not in {"prepare_db", "dense"}:
        validate_quant_database(
            quant_db,
            config,
            allow_unmanifested=args.allow_unmanifested_db,
        )
    storage_preflight = None
    memory_preflight = None
    if args.method == "prepare_db":
        resolved_output = output_dir.resolve()
        prepare_paths = [("output", resolved_output), ("database", quant_db.resolve())]
        if activation_cache_dir is not None:
            prepare_paths.append(("activation cache", activation_cache_dir.resolve()))
        for index, (left_name, left_path) in enumerate(prepare_paths):
            for right_name, right_path in prepare_paths[index + 1 :]:
                if (
                    left_path == right_path
                    or left_path in right_path.parents
                    or right_path in left_path.parents
                ):
                    raise ValueError(
                        "Prepare-db output paths must be separate and non-nested: "
                        f"{left_name}={left_path}, {right_name}={right_path}."
                    )
        if quant_db.exists():
            raise FileExistsError(
                f"Refusing to reuse existing quantization database target: {quant_db}"
            )
        if activation_cache_dir is not None and activation_cache_dir.exists():
            raise FileExistsError(
                f"Refusing to reuse existing activation-cache target: {activation_cache_dir}"
            )
        memory_preflight = cgroup_memory_snapshot()
        effective_processes = (
            config["quant_database"]["torchrun_processes"]
            if args.torchrun_processes is None
            else args.torchrun_processes
        )
        validate_prepare_db_memory_mode(
            args.database_memory_mode,
            effective_processes,
            memory_preflight,
        )
        storage_preflight = prepare_db_storage_preflight(
            quant_db,
            activation_cache_dir,
        )
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse existing experiment directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    resolved = {
        "source_config": str(config_path),
        "resolved_at": utc_now(),
        "git_commit": git_commit(),
        "method": args.method,
        "attempt_id": attempt_id,
        "seed": args.seed,
        "quant_database": str(quant_db),
        "output_dir": str(output_dir),
        "configuration": config,
        "runtime_overrides": {
            "configured_torchrun_processes": (
                config["quant_database"]["torchrun_processes"]
                if args.method == "prepare_db"
                else None
            ),
            "effective_torchrun_processes": (
                (
                    config["quant_database"]["torchrun_processes"]
                    if args.torchrun_processes is None
                    else args.torchrun_processes
                )
                if args.method == "prepare_db"
                else None
            ),
            "database_memory_mode": (
                args.database_memory_mode
                if args.method == "prepare_db"
                else None
            ),
            "activation_cache_dir": (
                str(activation_cache_dir)
                if activation_cache_dir is not None
                else None
            ),
            "resume_checkpoint": (
                str(resume_checkpoint)
                if resume_checkpoint is not None
                else None
            ),
        },
        "storage_preflight": storage_preflight,
        "memory_preflight": memory_preflight,
        "expected_search_compute": expected_search_compute(config, args.method),
    }
    write_json(output_dir / "resolved_config.json", resolved)
    with (output_dir / "command.sh").open("x", encoding="utf-8") as command_handle:
        command_handle.write(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"cd {shlex.quote(str(REPO_ROOT))}\n"
            f"exec {shlex.join(command)}\n"
        )
        command_handle.flush()
        os.fsync(command_handle.fileno())
    os.chmod(output_dir / "command.sh", 0o755)

    start_wall = time.time()
    start_monotonic = time.monotonic()
    write_json(
        output_dir / "launcher_status.json",
        {
            "status": "running",
            "attempt_id": attempt_id,
            "timestamp": utc_now(),
            "exit_code": None,
        },
    )
    try:
        exit_code, resource_summary = run_and_tee(
            command,
            output_dir / "run.log",
            output_dir / "resource_samples.jsonl",
            attempt_id=attempt_id,
        )
    except BaseException as error:
        runtime_seconds = time.monotonic() - start_monotonic
        write_json(
            output_dir / "runtime.json",
            {
                "attempt_id": attempt_id,
                "started_unix_seconds": start_wall,
                "ended_unix_seconds": time.time(),
                "runtime_seconds": runtime_seconds,
                "exit_code": None,
                "launcher_exception": repr(error),
            },
        )
        write_json(
            output_dir / "launcher_status.json",
            {
                "status": "failed",
                "phase": "launch_or_execution",
                "attempt_id": attempt_id,
                "exit_code": None,
                "exception": repr(error),
                "timestamp": utc_now(),
            },
        )
        raise
    child_runtime_seconds = time.monotonic() - start_monotonic
    runtime = {
        "attempt_id": attempt_id,
        "started_unix_seconds": start_wall,
        "ended_unix_seconds": time.time(),
        "runtime_seconds": child_runtime_seconds,
        "child_runtime_seconds": child_runtime_seconds,
        "exit_code": exit_code,
        "resource_summary": resource_summary,
        "resource_samples_path": str(output_dir / "resource_samples.jsonl"),
        "postflight_status": "pending" if exit_code == 0 else "not_run",
    }
    write_json(output_dir / "runtime.json", runtime)
    if exit_code != 0:
        write_json(
            output_dir / "launcher_status.json",
            {
                "status": "failed",
                "phase": "child_process",
                "attempt_id": attempt_id,
                "exit_code": exit_code,
                "timestamp": utc_now(),
            },
        )
        return exit_code

    write_json(
        output_dir / "launcher_status.json",
        {
            "status": "validating",
            "attempt_id": attempt_id,
            "exit_code": exit_code,
            "timestamp": utc_now(),
        },
    )
    try:
        if args.method == "prepare_db":
            validate_quant_database(
                quant_db,
                config,
                allow_unmanifested=False,
                expected_attempt_id=attempt_id,
            )
        if args.method in {"dense", "uniform3"}:
            log_text = (output_dir / "run.log").read_text(encoding="utf-8")
            metrics = parse_perplexities(log_text)
            missing = set(config["search"]["eval_datasets"]) - set(metrics)
            if missing:
                raise RuntimeError(
                    f"Completed evaluation is missing datasets: {sorted(missing)}"
                )
            write_json(
                output_dir / "run_summary.json",
                eval_summary(
                    args.method,
                    config,
                    args.seed,
                    child_runtime_seconds,
                    metrics,
                    output_dir,
                    quant_db,
                ),
            )
        elif args.method in {"quant_only", "joint"}:
            summary_path = output_dir / "run_summary.json"
            summary = validate_search_run_summary(
                summary_path,
                args.method,
                config,
                args.seed,
            )
            summary["launcher_attempt_id"] = attempt_id
            summary["launcher_wall_clock"] = runtime
            summary["resolved_config_path"] = str(
                output_dir / "resolved_config.json"
            )
            write_json(summary_path, summary)
    except BaseException as error:
        runtime["ended_unix_seconds"] = time.time()
        runtime["runtime_seconds"] = time.monotonic() - start_monotonic
        runtime["postflight_status"] = "failed"
        runtime["postflight_exception"] = repr(error)
        write_json(output_dir / "runtime.json", runtime)
        write_json(
            output_dir / "launcher_status.json",
            {
                "status": "failed",
                "phase": "postflight_validation",
                "attempt_id": attempt_id,
                "exit_code": exit_code,
                "exception": repr(error),
                "timestamp": utc_now(),
            },
        )
        raise

    runtime["ended_unix_seconds"] = time.time()
    runtime["runtime_seconds"] = time.monotonic() - start_monotonic
    runtime["postflight_status"] = "passed"
    write_json(output_dir / "runtime.json", runtime)
    write_json(
        output_dir / "launcher_status.json",
        {
            "status": "completed",
            "attempt_id": attempt_id,
            "exit_code": exit_code,
            "timestamp": utc_now(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
