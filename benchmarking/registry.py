from __future__ import annotations

from baselines import ALL_BASELINE_IDS, get_baseline_spec, normalize_baseline_id

from .types import MethodSpec

DCT_SCHUR_ID = "dct_schur_invariant_relational"


def proposal_spec() -> MethodSpec:
    return MethodSpec(
        method_id=DCT_SCHUR_ID,
        method_kind="proposal",
        display_name="Invariant-Relational DCT-Schur",
        blindness_tier="key_assisted_blind",
        fidelity_tier="proposal",
        requires_original_host=False,
        common_4096_payload=True,
        payload_size=64,
        comparison_group="proposal_key_assisted_4096",
        cover_dependent_key=True,
    )


def baseline_spec(method_id: str) -> MethodSpec:
    native = get_baseline_spec(normalize_baseline_id(method_id))
    payload_size = 64 if native.common_4096_payload else 16
    return MethodSpec(
        method_id=native.canonical_id,
        method_kind="baseline",
        display_name=native.display_name,
        blindness_tier=native.blindness_tier,
        fidelity_tier=native.fidelity_tier,
        requires_original_host=native.requires_original_host,
        common_4096_payload=native.common_4096_payload,
        payload_size=payload_size,
        comparison_group=f"baseline_{native.blindness_tier}_{payload_size}x{payload_size}",
        cover_dependent_key=native.blindness_tier != "blind",
    )


def get_method_spec(method_id: str) -> MethodSpec:
    normalized = str(method_id).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {DCT_SCHUR_ID, "dct_schur", "schur", "proposal"}:
        return proposal_spec()
    return baseline_spec(normalized)


def all_method_specs() -> tuple[MethodSpec, ...]:
    return (proposal_spec(),) + tuple(baseline_spec(value) for value in ALL_BASELINE_IDS)


def _selected_ids(selector: str) -> tuple[str, ...]:
    key = selector.strip().lower()
    specs = all_method_specs()
    if key == "all":
        return tuple(spec.method_id for spec in specs)
    if key in {"proposal", "proposals", "dct_schur", "schur"}:
        return (DCT_SCHUR_ID,)
    if key in {"baseline", "baselines"}:
        return tuple(ALL_BASELINE_IDS)
    tier_aliases = {
        "blind": "blind",
        "strict_blind": "blind",
        "semi_blind": "semi_blind",
        "semiblind": "semi_blind",
        "key_assisted": "key_assisted_blind",
        "key_assisted_blind": "key_assisted_blind",
        "non_blind": "non_blind",
        "nonblind": "non_blind",
    }
    if key in tier_aliases:
        tier = tier_aliases[key]
        return tuple(spec.method_id for spec in specs if spec.blindness_tier == tier)
    if key in {"common_4096", "payload_4096"}:
        return tuple(spec.method_id for spec in specs if spec.common_4096_payload)
    return tuple(part.strip() for part in selector.split(",") if part.strip())


def resolve_methods(selector: str | list[str] | tuple[str, ...]) -> tuple[MethodSpec, ...]:
    raw = _selected_ids(selector) if isinstance(selector, str) else tuple(selector)
    output: list[MethodSpec] = []
    seen: set[str] = set()
    for method_id in raw:
        spec = get_method_spec(method_id)
        if spec.method_id not in seen:
            output.append(spec)
            seen.add(spec.method_id)
    if not output:
        raise ValueError("No benchmark methods were selected")
    return tuple(output)


__all__ = [
    "DCT_SCHUR_ID",
    "proposal_spec",
    "baseline_spec",
    "get_method_spec",
    "all_method_specs",
    "resolve_methods",
]
