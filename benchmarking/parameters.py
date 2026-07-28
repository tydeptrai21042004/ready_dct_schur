from __future__ import annotations

import json
from pathlib import Path

from .types import BaselineParameters


def load_baseline_parameters(path: str | Path | None) -> dict[str, BaselineParameters]:
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    raw = json.loads(source.read_text(encoding="utf-8"))
    output: dict[str, BaselineParameters] = {}
    for method_id, values in raw.items():
        row = dict(values or {})
        output[str(method_id)] = BaselineParameters(
            repeat=row.get("repeat", 1),
            step=row.get("step"),
            method_params=dict(row.get("method_params", {})),
        )
    return output


__all__ = ["load_baseline_parameters"]
