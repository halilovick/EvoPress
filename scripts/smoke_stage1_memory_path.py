#!/usr/bin/env python3
"""Non-quantizing smoke test for the constrained-memory Stage-1 path.

The script deliberately creates only a fresh temporary activation-cache tree
under a caller-supplied scratch parent. It never opens or writes a quantization
database.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import tempfile
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODEL_NAME = "mistralai/Mistral-7B-v0.3"
SEQUENCE_LENGTH = 8192
EXPECTED_DENSE_PARAMETERS = 7_248_023_552
EXPECTED_ARCHITECTURE = {
    "model_type": "mistral",
    "vocab_size": 32_768,
    "hidden_size": 4_096,
    "intermediate_size": 14_336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
}
SCRATCH_PREFIX = ".evopress-stage1-memory-smoke-"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test direct-GPU Mistral loading and one disk-backed "
            "activation propagation without quantizing weights."
        )
    )
    parser.add_argument(
        "--scratch-parent",
        type=Path,
        required=True,
        help=(
            "Existing writable directory in which one fresh temporary smoke "
            "subdirectory will be created and subsequently removed."
        ),
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-name-or-path",
        default=MODEL_NAME,
        help="Defaults to the exact Stage-1 Mistral checkpoint.",
    )
    args = parser.parse_args(argv)
    if args.device_index < 0:
        parser.error("--device-index must be non-negative.")
    return args


def validate_scratch_parent(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(
            f"Scratch parent must already exist and be a directory: {resolved}"
        )
    if not os.access(resolved, os.W_OK | os.X_OK):
        raise PermissionError(f"Scratch parent is not writable: {resolved}")
    return resolved


@contextmanager
def owned_scratch_directory(parent: Path) -> Iterator[Path]:
    """Yield one fresh child and let TemporaryDirectory remove only that child."""

    resolved_parent = validate_scratch_parent(parent)
    with tempfile.TemporaryDirectory(
        prefix=SCRATCH_PREFIX,
        dir=str(resolved_parent),
    ) as value:
        scratch = Path(value).resolve()
        if scratch.parent != resolved_parent or not scratch.name.startswith(
            SCRATCH_PREFIX
        ):
            raise RuntimeError(f"Unexpected owned scratch path: {scratch}")
        yield scratch


def emit_event(event: str, **values: Any) -> None:
    print(
        json.dumps(
            {"schema_version": 1, "event": event, **values},
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def installed_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "bytes": int(tensor.numel() * tensor.element_size()),
        "contiguous": bool(tensor.is_contiguous()),
    }


def _tensor_inventory(named_tensors) -> dict[str, Any]:
    devices: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    total_numel = 0
    total_bytes = 0
    tensor_count = 0
    for _, tensor in named_tensors:
        tensor_count += 1
        numel = int(tensor.numel())
        size_bytes = numel * tensor.element_size()
        devices[str(tensor.device)] += size_bytes
        dtypes[str(tensor.dtype).removeprefix("torch.")] += size_bytes
        total_numel += numel
        total_bytes += size_bytes
    return {
        "tensor_count": tensor_count,
        "numel": total_numel,
        "bytes": total_bytes,
        "bytes_by_device": dict(sorted(devices.items())),
        "bytes_by_dtype": dict(sorted(dtypes.items())),
    }


def assert_model_residency(
    model: nn.Module,
    expected_device: torch.device,
    *,
    expected_parameter_dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    expected_device = torch.device(expected_device)
    parameters = list(model.named_parameters())
    buffers = list(model.named_buffers())
    wrong_parameter_devices = [
        name for name, value in parameters if value.device != expected_device
    ]
    wrong_buffer_devices = [
        name for name, value in buffers if value.device != expected_device
    ]
    wrong_parameter_dtypes = [
        f"{name}:{value.dtype}"
        for name, value in parameters
        if value.is_floating_point() and value.dtype != expected_parameter_dtype
    ]
    if wrong_parameter_devices or wrong_buffer_devices:
        raise RuntimeError(
            "Model is not resident entirely on the requested device: "
            f"parameter_mismatches={wrong_parameter_devices}, "
            f"buffer_mismatches={wrong_buffer_devices}."
        )
    if wrong_parameter_dtypes:
        raise RuntimeError(
            "Model floating-point parameters are not uniformly FP16: "
            f"{wrong_parameter_dtypes}."
        )
    return {
        "expected_device": str(expected_device),
        "expected_parameter_dtype": str(expected_parameter_dtype).removeprefix(
            "torch."
        ),
        "parameters": _tensor_inventory(parameters),
        "buffers": _tensor_inventory(buffers),
    }


def validate_mistral_architecture(config: Any) -> dict[str, Any]:
    observed = {
        "model_type": getattr(config, "model_type", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
    }
    attention_heads = observed["num_attention_heads"]
    hidden_size = observed["hidden_size"]
    observed["head_dim"] = getattr(config, "head_dim", None) or (
        hidden_size // attention_heads
        if isinstance(hidden_size, int)
        and isinstance(attention_heads, int)
        and attention_heads > 0
        else None
    )
    mismatches = {
        key: {"expected": expected, "actual": observed.get(key)}
        for key, expected in EXPECTED_ARCHITECTURE.items()
        if observed.get(key) != expected
    }
    max_positions = getattr(config, "max_position_embeddings", None)
    if not isinstance(max_positions, int) or max_positions < SEQUENCE_LENGTH:
        mismatches["max_position_embeddings"] = {
            "expected_minimum": SEQUENCE_LENGTH,
            "actual": max_positions,
        }
    if mismatches:
        raise RuntimeError(f"Unexpected Mistral architecture: {mismatches}")
    observed["max_position_embeddings"] = max_positions
    observed["sliding_window"] = getattr(config, "sliding_window", None)
    return observed


def validate_captured_mistral_inputs(
    input_args: Any,
    input_kwargs: Any,
    *,
    sequence_length: int,
    hidden_size: int,
    head_dim: int,
) -> dict[str, Any]:
    if not isinstance(input_args, (tuple, list)) or len(input_args) != 1:
        raise RuntimeError(
            "Expected exactly one positional Mistral block input, got "
            f"{type(input_args).__name__} with length "
            f"{len(input_args) if hasattr(input_args, '__len__') else 'unknown'}."
        )
    if not isinstance(input_kwargs, dict):
        raise RuntimeError("Mistral block kwargs were not stored as a dictionary.")

    hidden_states = input_args[0]
    if not isinstance(hidden_states, torch.Tensor):
        raise RuntimeError("Captured hidden states are not a Tensor.")
    expected_hidden_shape = [1, sequence_length, hidden_size]
    if list(hidden_states.shape) != expected_hidden_shape:
        raise RuntimeError(
            f"Captured hidden shape {list(hidden_states.shape)} != "
            f"{expected_hidden_shape}."
        )
    if hidden_states.dtype != torch.float16 or hidden_states.device.type != "cpu":
        raise RuntimeError(
            "Captured hidden states must be CPU FP16, got "
            f"{hidden_states.device}/{hidden_states.dtype}."
        )

    required_keys = {
        "attention_mask",
        "position_ids",
        "past_key_values",
        "use_cache",
        "cache_position",
        "position_embeddings",
    }
    missing = required_keys - set(input_kwargs)
    if missing:
        raise RuntimeError(f"Captured Mistral kwargs are missing: {sorted(missing)}")
    if input_kwargs["attention_mask"] is not None:
        raise RuntimeError("Unpadded FlashAttention 2 input should have no mask tensor.")
    if input_kwargs["past_key_values"] is not None or input_kwargs["use_cache"] is not False:
        raise RuntimeError("Smoke propagation requires a disabled KV cache.")

    position_ids = input_kwargs["position_ids"]
    cache_position = input_kwargs["cache_position"]
    if not isinstance(position_ids, torch.Tensor) or list(position_ids.shape) != [
        1,
        sequence_length,
    ]:
        raise RuntimeError("Captured position_ids have the wrong shape or type.")
    if not isinstance(cache_position, torch.Tensor) or list(
        cache_position.shape
    ) != [sequence_length]:
        raise RuntimeError("Captured cache_position has the wrong shape or type.")
    if (
        position_ids.dtype != torch.int64
        or cache_position.dtype != torch.int64
        or position_ids.device.type != "cpu"
        or cache_position.device.type != "cpu"
        or not torch.equal(position_ids, cache_position.unsqueeze(0))
    ):
        raise RuntimeError("Captured Mistral position metadata is inconsistent.")

    position_embeddings = input_kwargs["position_embeddings"]
    if not isinstance(position_embeddings, tuple) or len(position_embeddings) != 2:
        raise RuntimeError("Captured position_embeddings must be a (cos, sin) tuple.")
    for value in position_embeddings:
        if (
            not isinstance(value, torch.Tensor)
            or list(value.shape) != [1, sequence_length, head_dim]
            or value.dtype != torch.float16
            or value.device.type != "cpu"
        ):
            raise RuntimeError(
                "Captured RoPE tensors have an unexpected shape, dtype, or device."
            )

    return {
        "input_container": type(input_args).__name__,
        "kwarg_keys": sorted(input_kwargs),
        "hidden_states": tensor_summary(hidden_states),
        "position_ids": tensor_summary(position_ids),
        "cache_position": tensor_summary(cache_position),
        "position_embeddings": [
            tensor_summary(value) for value in position_embeddings
        ],
    }


def cuda_memory_snapshot(device: torch.device) -> dict[str, int]:
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def run_smoke(args: argparse.Namespace, scratch: Path) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM
    from transformers.utils import is_flash_attn_2_available

    from src.activation_cache import DiskActivationCache
    from src.calibration_utils import calibration_token_digest
    from src.common_utils import fix_seed, maybe_first_element, to
    from src.memory_utils import cgroup_memory_snapshot, release_cpu_memory
    from src.model_utils import ForwardInterrupt, InputCollector, get_layers
    from src.quantizer import Quantizer
    from src.run_reporting import get_git_commit

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if args.device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested cuda:{args.device_index}, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are visible."
        )
    if not is_flash_attn_2_available():
        raise RuntimeError("Transformers cannot use FlashAttention 2.")

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)
    fix_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    attempt_id = uuid.uuid4().hex
    cache_path = scratch / "activation-cache"
    timings: dict[str, float] = {}
    cgroup_memory = {"before_model_load": cgroup_memory_snapshot()}
    cuda_memory = {"before_model_load": cuda_memory_snapshot(device)}

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map={"": str(device)},
    )
    model.eval()
    timings["model_load_seconds"] = time.perf_counter() - load_started
    architecture = validate_mistral_architecture(model.config)
    residency = assert_model_residency(model, device)
    if residency["parameters"]["numel"] != EXPECTED_DENSE_PARAMETERS:
        raise RuntimeError(
            "Unexpected dense parameter count: "
            f"{residency['parameters']['numel']} != {EXPECTED_DENSE_PARAMETERS}."
        )
    cgroup_memory["after_model_load"] = cgroup_memory_snapshot()
    cuda_memory["after_model_load"] = cuda_memory_snapshot(device)
    emit_event(
        "model_loaded",
        model=args.model_name_or_path,
        architecture=architecture,
        residency=residency,
        cgroup_memory=cgroup_memory["after_model_load"],
        cuda_memory=cuda_memory["after_model_load"],
    )

    cache = DiskActivationCache(
        cache_path,
        drop_file_cache=True,
        attempt_id=attempt_id,
    )
    layers = get_layers(model)
    original_first_block = layers[0]
    collector = InputCollector(
        original_first_block,
        cpu_offload=True,
        input_cache=cache,
    )
    original_use_cache = model.config.use_cache
    model.config.use_cache = False
    layers[0] = collector
    input_ids = (
        torch.arange(SEQUENCE_LENGTH, dtype=torch.int64, device=device)
        .remainder(int(model.config.vocab_size))
        .unsqueeze(0)
    )

    collect_started = time.perf_counter()
    interrupted = False
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    except ForwardInterrupt:
        interrupted = True
    finally:
        layers[0] = original_first_block
    if not interrupted or len(cache) != 1:
        raise RuntimeError(
            "InputCollector did not capture exactly one interrupted Mistral record."
        )
    cache.finish_collection()
    timings["input_collection_seconds"] = time.perf_counter() - collect_started
    del input_ids
    release_cpu_memory()

    input_args, input_kwargs = cache.load(0)
    captured_summary = validate_captured_mistral_inputs(
        input_args,
        input_kwargs,
        sequence_length=SEQUENCE_LENGTH,
        hidden_size=EXPECTED_ARCHITECTURE["hidden_size"],
        head_dim=EXPECTED_ARCHITECTURE["head_dim"],
    )
    cgroup_memory["after_collection"] = cgroup_memory_snapshot()
    cuda_memory["after_collection"] = cuda_memory_snapshot(device)
    emit_event(
        "activation_collected",
        captured=captured_summary,
        cache=cache.summary(),
        cgroup_memory=cgroup_memory["after_collection"],
        cuda_memory=cuda_memory["after_collection"],
    )

    propagate_started = time.perf_counter()
    with torch.inference_mode():
        block_output = original_first_block(
            *to(input_args, device=device),
            **to(input_kwargs, device=device),
        )
    block_output = maybe_first_element(block_output)
    if (
        not isinstance(block_output, torch.Tensor)
        or list(block_output.shape)
        != [1, SEQUENCE_LENGTH, EXPECTED_ARCHITECTURE["hidden_size"]]
        or block_output.dtype != torch.float16
        or block_output.device != device
        or not bool(torch.isfinite(block_output).all())
    ):
        raise RuntimeError("First Mistral block produced an invalid output tensor.")
    block_output_cpu = block_output.cpu()
    output_digest = calibration_token_digest([block_output_cpu])
    input_args, input_kwargs = Quantizer._replace_hidden_state(
        input_args,
        input_kwargs,
        block_output_cpu,
    )
    cache.replace(0, input_args, input_kwargs)
    cache.mark_block_complete(1, len(layers))
    reloaded_args, reloaded_kwargs = cache.load(0)
    propagated_summary = validate_captured_mistral_inputs(
        reloaded_args,
        reloaded_kwargs,
        sequence_length=SEQUENCE_LENGTH,
        hidden_size=EXPECTED_ARCHITECTURE["hidden_size"],
        head_dim=EXPECTED_ARCHITECTURE["head_dim"],
    )
    if not torch.equal(reloaded_args[0], block_output_cpu):
        raise RuntimeError("Reloaded first-block output is not bit-exact.")
    if calibration_token_digest([reloaded_args[0]]) != output_digest:
        raise RuntimeError("Reloaded first-block output digest changed.")
    timings["first_block_propagation_seconds"] = (
        time.perf_counter() - propagate_started
    )
    cgroup_memory["after_first_block"] = cgroup_memory_snapshot()
    cuda_memory["after_first_block"] = cuda_memory_snapshot(device)
    cache_manifest = json.loads(
        (cache_path / DiskActivationCache.MANIFEST_NAME).read_text(
            encoding="utf-8"
        )
    )
    expected_cache_manifest = {
        "attempt_id": attempt_id,
        "status": "processing",
        "sample_count": 1,
        "completed_blocks": 1,
        "total_blocks": len(layers),
    }
    cache_manifest_mismatches = {
        key: {"expected": expected, "actual": cache_manifest.get(key)}
        for key, expected in expected_cache_manifest.items()
        if cache_manifest.get(key) != expected
    }
    if cache_manifest_mismatches:
        raise RuntimeError(
            "One-block activation-cache manifest is inconsistent: "
            f"{cache_manifest_mismatches}"
        )

    model.config.use_cache = original_use_cache
    result = {
        "status": "passed",
        "attempt_id": attempt_id,
        "git_commit": get_git_commit(REPO_ROOT),
        "model_name": args.model_name_or_path,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "seed": args.seed,
        "sequence_length": SEQUENCE_LENGTH,
        "attention_implementation": "flash_attention_2",
        "dtype": "float16",
        "low_cpu_mem_usage": True,
        "architecture": architecture,
        "residency": residency,
        "captured": captured_summary,
        "propagated": propagated_summary,
        "propagated_hidden_digest_algorithm": (
            "sha256-v1-dtype-shape-boundaries"
        ),
        "propagated_hidden_digest": output_digest,
        "cache_summary": cache.summary(),
        "cache_manifest": cache_manifest,
        "cgroup_memory": cgroup_memory,
        "cuda_memory": cuda_memory,
        "timings": timings,
        "software_versions": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "transformers": installed_version("transformers"),
            "flash_attn": installed_version("flash-attn"),
        },
    }

    del block_output
    del block_output_cpu
    del input_args, input_kwargs, reloaded_args, reloaded_kwargs
    del collector, original_first_block, layers, model
    release_cpu_memory()
    torch.cuda.empty_cache()
    cgroup_memory["after_release"] = cgroup_memory_snapshot()
    cuda_memory["after_release"] = cuda_memory_snapshot(device)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scratch_parent = validate_scratch_parent(args.scratch_parent)
    started = time.perf_counter()
    scratch_path: Path | None = None
    try:
        with owned_scratch_directory(scratch_parent) as scratch:
            scratch_path = scratch
            emit_event(
                "started",
                scratch_parent=str(scratch_parent),
                owned_scratch=str(scratch),
                model=args.model_name_or_path,
                sequence_length=SEQUENCE_LENGTH,
                seed=args.seed,
            )
            result = run_smoke(args, scratch)
    except BaseException as error:
        emit_event(
            "failed",
            error_type=type(error).__name__,
            error=str(error),
            owned_scratch=str(scratch_path) if scratch_path is not None else None,
            scratch_removed=(
                not scratch_path.exists() if scratch_path is not None else None
            ),
            elapsed_seconds=time.perf_counter() - started,
        )
        raise

    result["owned_scratch"] = str(scratch_path)
    result["scratch_removed"] = not scratch_path.exists()
    result["elapsed_seconds"] = time.perf_counter() - started
    emit_event("completed", **result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
