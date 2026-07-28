from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(result: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return target


def write_summary_csv(result: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(result.get("summaries", []))
    fields = list(rows[0]) if rows else ["method_id"]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_attack_csv(result: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for trial in result.get("trials", []):
        for attack in trial.get("attacks", []):
            metrics = attack.get("watermark_metrics", {})
            rows.append(
                {
                    "method_id": trial.get("method_id"),
                    "method_kind": trial.get("method_kind"),
                    "blindness_tier": trial.get("blindness_tier"),
                    "host_id": trial.get("host_id"),
                    "payload_id": trial.get("payload_id"),
                    "payload_bits": trial.get("payload_bits"),
                    "attack_id": attack.get("attack_id"),
                    "category": attack.get("category"),
                    "severity": attack.get("severity"),
                    "status": attack.get("status"),
                    "nc": metrics.get("nc"),
                    "ncc": metrics.get("ncc"),
                    "ber": metrics.get("ber"),
                    "bit_accuracy": metrics.get("bit_accuracy"),
                }
            )
    fields = list(rows[0]) if rows else ["method_id"]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return target


__all__ = ["write_json", "write_summary_csv", "write_attack_csv"]
