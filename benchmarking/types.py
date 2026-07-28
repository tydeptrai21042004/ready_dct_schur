from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MethodKind = Literal["proposal", "baseline"]


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    method_kind: MethodKind
    display_name: str
    blindness_tier: str
    fidelity_tier: str
    requires_original_host: bool
    common_4096_payload: bool
    payload_size: int
    comparison_group: str
    cover_dependent_key: bool = False


@dataclass(frozen=True)
class BaselineParameters:
    repeat: int | str = 1
    step: float | None = None
    method_params: dict[str, Any] = field(default_factory=dict)


__all__ = ["MethodSpec", "BaselineParameters", "MethodKind"]
