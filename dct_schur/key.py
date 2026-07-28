from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .constants import METHOD_ID


@dataclass
class SchurKey:
    host_shape: tuple[int, int, int]
    payload_shape: tuple[int, int]
    config: dict[str, Any]
    spectral_reference: list[float]
    arnold_period: int
    payload_coset_flip_b64: str = ""
    active_shape: tuple[int, int] | None = None
    replica_count: int = 1
    key_version: int = 5

    @property
    def method_id(self) -> str:
        return METHOD_ID

    @property
    def blindness_class(self) -> str:
        return "key_assisted_blind"

    @property
    def watermark_shape(self) -> tuple[int, int]:
        """Compatibility alias for earlier keys."""
        return self.payload_shape

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SchurKey":
        raw = dict(value)
        if "payload_shape" not in raw and "watermark_shape" in raw:
            raw["payload_shape"] = raw.pop("watermark_shape")
        for field in ("host_shape", "payload_shape", "active_shape"):
            if raw.get(field) is not None:
                raw[field] = tuple(int(x) for x in raw[field])
        return cls(**raw)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SchurKey":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = ["SchurKey"]
