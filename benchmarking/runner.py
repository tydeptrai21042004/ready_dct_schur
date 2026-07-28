from __future__ import annotations

import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from evaluation.attacks import AttackConfig, apply_attack
from evaluation.metrics import image_quality_metrics, watermark_metrics

from .adapters import MethodAdapter
from .parameters import load_baseline_parameters
from .registry import resolve_methods
from .types import BaselineParameters, MethodSpec


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _key_size_bytes(key: Any) -> int | None:
    try:
        return len(pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return None


def prepare_payload(payload: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(payload)
    if arr.ndim != 2:
        raise ValueError(f"Payload must be two-dimensional, got {arr.shape}")
    threshold = 0 if arr.size and float(np.max(arr)) <= 1.0 else 127
    binary = np.where(arr > threshold, 255, 0).astype(np.uint8)
    if binary.shape == (size, size):
        return binary
    resized = Image.fromarray(binary, mode="L").resize((size, size), Image.Resampling.NEAREST)
    return np.where(np.asarray(resized) > 127, 255, 0).astype(np.uint8)


def evaluate_method_trial(
    adapter: MethodAdapter,
    host: np.ndarray,
    source_payload: np.ndarray,
    attacks: Iterable[AttackConfig],
    *,
    host_id: str,
    payload_id: str,
    seed: int = 2026,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    spec = adapter.spec
    host_u8 = np.asarray(host, dtype=np.uint8)
    payload = prepare_payload(source_payload, spec.payload_size)
    result: dict[str, Any] = {
        "method_id": spec.method_id,
        "method_kind": spec.method_kind,
        "display_name": spec.display_name,
        "blindness_tier": spec.blindness_tier,
        "fidelity_tier": spec.fidelity_tier,
        "comparison_group": spec.comparison_group,
        "requires_original_host": spec.requires_original_host,
        "common_4096_payload": spec.common_4096_payload,
        "payload_shape": list(payload.shape),
        "payload_bits": int(payload.size),
        "host_id": host_id,
        "payload_id": payload_id,
        "seed": int(seed),
        "configuration": adapter.configuration_summary(seed),
        "status": "ok",
        "attacks": [],
    }

    try:
        start = time.perf_counter()
        marked, key, embedding_metadata = adapter.embed(host_u8, payload, seed=seed)
        embed_seconds = time.perf_counter() - start
        clean_start = time.perf_counter()
        clean_recovered, clean_metadata = adapter.extract(
            marked,
            key,
            original_host=host_u8 if spec.requires_original_host else None,
        )
        clean_extract_seconds = time.perf_counter() - clean_start
        result.update(
            {
                "embed_seconds": float(embed_seconds),
                "clean_extract_seconds": float(clean_extract_seconds),
                "key_size_bytes": _key_size_bytes(key),
                "embedding_metrics": image_quality_metrics(host_u8, marked),
                "clean_watermark_metrics": watermark_metrics(payload, clean_recovered),
                "embedding_metadata": embedding_metadata,
                "clean_extract_metadata": clean_metadata,
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "embed_or_clean_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        if not continue_on_error:
            raise
        return _json_safe(result)

    for attack in attacks:
        row: dict[str, Any] = {
            "attack_id": attack.attack_id,
            "group": attack.group,
            "category": attack.category,
            "severity": attack.severity,
            "params": dict(attack.params),
            "status": "ok",
        }
        try:
            attack_start = time.perf_counter()
            attacked = apply_attack(marked, attack)
            row["attack_seconds"] = float(time.perf_counter() - attack_start)
            extract_start = time.perf_counter()
            recovered, metadata = adapter.extract(
                attacked,
                key,
                original_host=host_u8 if spec.requires_original_host else None,
            )
            row["extract_seconds"] = float(time.perf_counter() - extract_start)
            row["watermark_metrics"] = watermark_metrics(payload, recovered)
            row["attacked_image_metrics"] = image_quality_metrics(host_u8, attacked)
            row["extract_metadata"] = metadata
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            if not continue_on_error:
                raise
        result["attacks"].append(row)

    successful = [row for row in result["attacks"] if row["status"] == "ok"]
    result["attack_count"] = len(result["attacks"])
    result["successful_attacks"] = len(successful)
    result["mean_attacked_nc"] = (
        float(np.mean([row["watermark_metrics"]["nc"] for row in successful]))
        if successful
        else None
    )
    return _json_safe(result)


def run_benchmark_matrix(
    hosts: dict[str, np.ndarray],
    payloads: dict[str, np.ndarray],
    attacks: Iterable[AttackConfig],
    *,
    methods: str | list[str] | tuple[str, ...] = "all",
    seed: int = 2026,
    schur_config=None,
    baseline_parameters_path: str | Path | None = None,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    specs = resolve_methods(methods)
    parameters = load_baseline_parameters(baseline_parameters_path)
    attack_tuple = tuple(attacks)
    trials: list[dict[str, Any]] = []
    for spec in specs:
        adapter = MethodAdapter(
            spec,
            schur_config=schur_config,
            baseline_parameters=parameters.get(spec.method_id, BaselineParameters()),
        )
        for host_id, host in hosts.items():
            for payload_id, payload in payloads.items():
                trials.append(
                    evaluate_method_trial(
                        adapter,
                        host,
                        payload,
                        attack_tuple,
                        host_id=host_id,
                        payload_id=payload_id,
                        seed=seed,
                        continue_on_error=continue_on_error,
                    )
                )
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        method_trials = [trial for trial in trials if trial["method_id"] == spec.method_id]
        valid = [trial for trial in method_trials if trial["status"] == "ok"]
        attacked_values = [
            trial["mean_attacked_nc"]
            for trial in valid
            if trial.get("mean_attacked_nc") is not None
        ]
        psnr_values = [trial["embedding_metrics"]["psnr_db"] for trial in valid]
        clean_values = [trial["clean_watermark_metrics"]["nc"] for trial in valid]
        summaries.append(
            {
                "method_id": spec.method_id,
                "display_name": spec.display_name,
                "method_kind": spec.method_kind,
                "blindness_tier": spec.blindness_tier,
                "fidelity_tier": spec.fidelity_tier,
                "payload_bits": spec.payload_size * spec.payload_size,
                "trial_count": len(method_trials),
                "successful_trials": len(valid),
                "mean_psnr": float(np.mean(psnr_values)) if psnr_values else None,
                "mean_clean_nc": float(np.mean(clean_values)) if clean_values else None,
                "mean_attacked_nc": float(np.mean(attacked_values)) if attacked_values else None,
            }
        )
    return _json_safe(
        {
            "schema_version": 1,
            "seed": int(seed),
            "methods": [spec.method_id for spec in specs],
            "host_count": len(hosts),
            "payload_count": len(payloads),
            "attack_count": len(attack_tuple),
            "summaries": summaries,
            "trials": trials,
        }
    )


__all__ = ["prepare_payload", "evaluate_method_trial", "run_benchmark_matrix"]
