#!/usr/bin/env python3
"""Merge launcher-observed runtime and memory peaks into run_summary.json."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize a structured run summary with launcher-level measurements."
    )
    parser.add_argument("--summary", required=True, help="Path to run_summary.json.")
    parser.add_argument("--runtime-file", required=True, help="Launcher runtime.txt file.")
    parser.add_argument(
        "--memory-samples",
        default=None,
        help="Optional launcher memory_samples.csv file.",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def finite_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def sampled_memory_peaks(path: Path | None) -> tuple[float | None, float | None]:
    if path is None or not path.is_file():
        return None, None

    cpu_peak_gb: float | None = None
    gpu_peak_gb: float | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cpu_value = finite_float(row.get("cpu_memory_current_gb"))
            gpu_value = finite_float(row.get("gpu_memory_used_gb"))
            if cpu_value is not None:
                cpu_peak_gb = cpu_value if cpu_peak_gb is None else max(cpu_peak_gb, cpu_value)
            if gpu_value is not None:
                gpu_peak_gb = gpu_value if gpu_peak_gb is None else max(gpu_peak_gb, gpu_value)
    return cpu_peak_gb, gpu_peak_gb


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def finalize_summary(
    summary_path: Path,
    runtime_path: Path,
    memory_samples_path: Path | None,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runtime_values = parse_key_values(runtime_path)
    runtime_seconds = finite_float(runtime_values.get("runtime_seconds"))
    cpu_peak_gb, gpu_peak_gb = sampled_memory_peaks(memory_samples_path)

    final_metrics = summary["final_metrics"]
    process_runtime = final_metrics.get(
        "search_process_runtime_seconds",
        final_metrics.get("runtime_seconds"),
    )
    process_rss = final_metrics.get(
        "peak_process_rss_mb",
        final_metrics.get("peak_cpu_memory_mb"),
    )

    final_metrics["search_process_runtime_seconds"] = process_runtime
    if runtime_seconds is not None:
        final_metrics["runtime_seconds"] = runtime_seconds
    final_metrics["runtime_source"] = "launcher_wall_clock"

    final_metrics["peak_process_rss_mb"] = process_rss
    if cpu_peak_gb is not None:
        cgroup_peak_mb = cpu_peak_gb * 1024
        final_metrics["peak_cpu_cgroup_memory_mb"] = cgroup_peak_mb
        final_metrics["peak_cpu_memory_mb"] = cgroup_peak_mb
        final_metrics["peak_cpu_memory_source"] = "sampled_cgroup_memory.current"
    else:
        final_metrics["peak_cpu_cgroup_memory_mb"] = None
        final_metrics["peak_cpu_memory_source"] = "process_ru_maxrss"

    final_metrics["peak_gpu_device_used_mb"] = (
        gpu_peak_gb * 1024 if gpu_peak_gb is not None else None
    )
    final_metrics["peak_gpu_device_used_source"] = (
        "sampled_nvidia_smi" if gpu_peak_gb is not None else None
    )

    summary["launcher_finalized_at"] = utc_now()
    summary["artifacts"]["runtime_path"] = str(runtime_path)
    summary["artifacts"]["memory_samples_path"] = (
        str(memory_samples_path) if memory_samples_path is not None else None
    )
    summary["metric_definitions"]["peak_cpu_memory_mb"] = (
        "Peak sampled container cgroup memory when memory_samples.csv is available; "
        "otherwise peak process RSS."
    )
    summary["metric_definitions"]["peak_process_rss_mb"] = (
        "Peak resident set size for the Python search process."
    )
    summary["metric_definitions"]["peak_gpu_device_used_mb"] = (
        "Peak sampled total GPU device memory in use from nvidia-smi."
    )
    summary["metric_definitions"]["runtime_seconds"] = (
        "Launcher-measured model-command wall-clock runtime, including process startup."
    )

    write_json_atomic(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary_path = Path(args.summary)
    runtime_path = Path(args.runtime_file)
    memory_samples_path = Path(args.memory_samples) if args.memory_samples else None

    if not summary_path.is_file():
        raise ValueError(f"Summary file does not exist: {summary_path}")
    if not runtime_path.is_file():
        raise ValueError(f"Runtime file does not exist: {runtime_path}")

    summary = finalize_summary(summary_path, runtime_path, memory_samples_path)
    print(f"Finalized structured run summary: {summary_path}")
    print(f"Runtime seconds: {summary['final_metrics']['runtime_seconds']}")
    print(f"Peak CPU memory MB: {summary['final_metrics']['peak_cpu_memory_mb']}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
