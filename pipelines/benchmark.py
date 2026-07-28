from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from benchmarking.output import write_attack_csv, write_json, write_summary_csv
from benchmarking.runner import run_benchmark_matrix
from dct_schur import SchurConfig
from dct_schur.io import load_binary_plane, load_rgb
from evaluation.attacks import get_attack_suite

_IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def _collect_rgb(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_file():
        return {source.name: load_rgb(source)}
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = sorted(item for item in source.iterdir() if item.suffix.lower() in _IMAGE_SUFFIXES)
    if not files:
        raise ValueError(f"No host images found in {source}")
    return {item.name: load_rgb(item) for item in files}


def _collect_payloads(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_file():
        return {source.name: load_binary_plane(source, side=64)}
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = sorted(item for item in source.iterdir() if item.suffix.lower() in _IMAGE_SUFFIXES)
    if not files:
        raise ValueError(f"No payload images found in {source}")
    return {item.name: load_binary_plane(item, side=64) for item in files}


def run_comparison_benchmark(
    host_path: str | Path,
    payload_path: str | Path,
    output_json: str | Path,
    *,
    methods: str | list[str] | tuple[str, ...] = "all",
    output_summary_csv: str | Path | None = None,
    output_attack_csv: str | Path | None = None,
    attack_suite: str = "extended",
    config: SchurConfig | Mapping[str, Any] | None = None,
    baseline_parameters_path: str | Path | None = "configs/baseline_parameters.json",
    seed: int = 2026,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    schur_config = config if isinstance(config, SchurConfig) else SchurConfig.from_mapping(config or {})
    result = run_benchmark_matrix(
        _collect_rgb(host_path),
        _collect_payloads(payload_path),
        get_attack_suite(attack_suite),
        methods=methods,
        seed=seed,
        schur_config=schur_config,
        baseline_parameters_path=baseline_parameters_path,
        continue_on_error=continue_on_error,
    )
    result["attack_suite"] = attack_suite
    result["host_path"] = str(host_path)
    result["payload_path"] = str(payload_path)
    write_json(result, output_json)
    if output_summary_csv is not None:
        write_summary_csv(result, output_summary_csv)
    if output_attack_csv is not None:
        write_attack_csv(result, output_attack_csv)
    return result


def benchmark_image_payload(
    host_path: str | Path,
    payload_path: str | Path,
    output_json: str | Path,
    *,
    output_csv: str | Path | None = None,
    attack_suite: str = "extended",
    config: SchurConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Backward-compatible proposal-only benchmark wrapper."""
    return run_comparison_benchmark(
        host_path,
        payload_path,
        output_json,
        methods="dct_schur",
        output_attack_csv=output_csv,
        attack_suite=attack_suite,
        config=config,
    )


__all__ = ["run_comparison_benchmark", "benchmark_image_payload"]
