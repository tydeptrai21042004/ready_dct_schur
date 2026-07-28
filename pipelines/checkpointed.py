from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from benchmarking.registry import resolve_methods
from dct_schur import SchurConfig

from .benchmark import run_comparison_benchmark


def run_checkpointed_benchmark(
    host_path: str | Path,
    payload_path: str | Path,
    output_directory: str | Path,
    *,
    methods: str | list[str] | tuple[str, ...] = "all",
    attack_suite: str = "extended",
    config: SchurConfig | Mapping[str, Any] | None = None,
    baseline_parameters_path: str | Path | None = "configs/baseline_parameters.json",
    seed: int = 2026,
    resume: bool = True,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    """Run one method at a time and preserve completed method checkpoints."""
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = resolve_methods(methods)
    index: dict[str, Any] = {
        "schema_version": 1,
        "host_path": str(host_path),
        "payload_path": str(payload_path),
        "attack_suite": attack_suite,
        "seed": int(seed),
        "requested_methods": [spec.method_id for spec in specs],
        "completed": [],
        "failed": [],
        "checkpoints": {},
    }
    index_path = output_dir / "index.json"

    for position, spec in enumerate(specs, start=1):
        result_path = output_dir / f"{spec.method_id}.json"
        summary_path = output_dir / f"{spec.method_id}_summary.csv"
        attacks_path = output_dir / f"{spec.method_id}_attacks.csv"
        if resume and result_path.exists():
            try:
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                if existing.get("methods") == [spec.method_id]:
                    index["completed"].append(spec.method_id)
                    index["checkpoints"][spec.method_id] = str(result_path)
                    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
                    print(f"[{position}/{len(specs)}] resume {spec.method_id}", flush=True)
                    continue
            except Exception:
                pass

        print(f"[{position}/{len(specs)}] run {spec.method_id}", flush=True)
        try:
            result = run_comparison_benchmark(
                host_path,
                payload_path,
                result_path,
                methods=spec.method_id,
                output_summary_csv=summary_path,
                output_attack_csv=attacks_path,
                attack_suite=attack_suite,
                config=config,
                baseline_parameters_path=baseline_parameters_path,
                seed=seed,
                continue_on_error=continue_on_error,
            )
            successful = all(row.get("successful_trials", 0) > 0 for row in result.get("summaries", []))
            if successful:
                index["completed"].append(spec.method_id)
            else:
                index["failed"].append(spec.method_id)
            index["checkpoints"][spec.method_id] = str(result_path)
        except Exception as exc:
            index["failed"].append(spec.method_id)
            index["checkpoints"][spec.method_id] = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            if not continue_on_error:
                index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
                raise
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    index["status"] = "complete" if not index["failed"] else "complete_with_errors"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


__all__ = ["run_checkpointed_benchmark"]
