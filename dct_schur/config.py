from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .constants import METHOD_ID, SHORT_NAME


@dataclass(frozen=True)
class SchurConfig:
    """Configuration for the single DCT-Schur method.

    The defaults preserve the validated >50 dB embedding profile. Extraction
    adds witness-gated, robust consensus evidence without increasing distortion.
    """

    step: float = 9.0
    eta: float = 0.07
    seed: int = 2026
    arnold_iterations: int = 17
    closure_rounds: int = 2

    gain_normalization_enabled: bool = True
    gain_gamma: float = 0.75
    gain_clip: tuple[float, float] = (0.55, 1.45)
    payload_coset_optimization_enabled: bool = True

    confidence_floor: float = 0.10
    confidence_scale: float = 0.90
    confidence_power: float = 1.50

    # New identity-witness extraction: remove global gain, then use local
    # diagonal drift to reduce trust in damaged relational carriers.
    witness_gating_enabled: bool = True
    witness_log_sigma: float = 0.50
    witness_floor: float = 0.30

    # New robust aggregation: a bounded consensus term prevents one damaged
    # Schur observation from overpowering two agreeing observations.
    robust_consensus_enabled: bool = False
    consensus_median_weight: float = 0.70
    consensus_mean_weight: float = 0.30
    consensus_clip: float = 1.25
    erasure_threshold: float = 0.18

    map_lambda: float = 0.55
    map_iters: int = 20
    candidate_search_enabled: bool = True
    sharpness_factors: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 4.0)
    unsharp_candidates: tuple[tuple[float, int], ...] = (
        (0.5, 100), (1.0, 100), (1.0, 150),
        (1.0, 250), (1.5, 150), (2.0, 200),
    )
    candidate_agreement_weight: float = 0.50
    candidate_evidence_weight: float = 0.25
    candidate_confidence_weight: float = 0.10
    candidate_witness_weight: float = 0.15

    schur_lift: float = 128.0
    determinant_epsilon: float = 1e-12
    resolution_adaptive_replicas_enabled: bool = False
    max_payload_replicas: int = 8
    replica_step_exponent: float = 0.20
    minimum_host_side: int = 512

    def validated(self) -> "SchurConfig":
        if self.step <= 0:
            raise ValueError("step must be positive")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be in [0,1]")
        if self.closure_rounds < 1:
            raise ValueError("closure_rounds must be at least 1")
        if len(self.gain_clip) != 2 or not 0 < self.gain_clip[0] <= self.gain_clip[1]:
            raise ValueError("gain_clip must be an ordered positive pair")
        if self.confidence_floor < 0 or self.confidence_scale < 0:
            raise ValueError("confidence weights must be nonnegative")
        if self.confidence_power <= 0:
            raise ValueError("confidence_power must be positive")
        if self.witness_log_sigma <= 0:
            raise ValueError("witness_log_sigma must be positive")
        if not 0.0 <= self.witness_floor <= 1.0:
            raise ValueError("witness_floor must be in [0,1]")
        if self.consensus_median_weight < 0 or self.consensus_mean_weight < 0:
            raise ValueError("consensus weights must be nonnegative")
        if self.consensus_median_weight + self.consensus_mean_weight <= 0:
            raise ValueError("at least one consensus weight must be positive")
        if self.consensus_clip <= 0:
            raise ValueError("consensus_clip must be positive")
        if not 0.0 <= self.erasure_threshold <= 1.0:
            raise ValueError("erasure_threshold must be in [0,1]")
        candidate_total = (
            self.candidate_agreement_weight + self.candidate_evidence_weight
            + self.candidate_confidence_weight + self.candidate_witness_weight
        )
        if candidate_total <= 0:
            raise ValueError("candidate score weights must have positive sum")
        if self.map_lambda < 0 or self.map_iters < 0:
            raise ValueError("invalid MAP parameters")
        if self.schur_lift <= 0 or self.determinant_epsilon <= 0:
            raise ValueError("Schur lift and determinant epsilon must be positive")
        if self.max_payload_replicas < 1:
            raise ValueError("max_payload_replicas must be at least 1")
        if not 0.0 <= self.replica_step_exponent <= 0.5:
            raise ValueError("replica_step_exponent must be in [0,0.5]")
        if self.minimum_host_side < 512:
            raise ValueError("minimum_host_side must be at least 512")
        return self

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output.update({"method_id": METHOD_ID, "scientific_name": SHORT_NAME})
        return output

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "SchurConfig" | None) -> "SchurConfig":
        if value is None:
            return cls().validated()
        if isinstance(value, cls):
            return value.validated()
        raw = dict(value)
        aliases = {
            "schur_closure_iters": "closure_rounds",
            "direct_det_epsilon": "determinant_epsilon",
            "qr_map_lambda": "map_lambda",
            "qr_map_iters": "map_iters",
            "evidence_conf_power": "confidence_power",
        }
        for old, new in aliases.items():
            if old in raw and new not in raw:
                raw[new] = raw.pop(old)
        raw.pop("base_config", None)
        raw.pop("schur_step", None)
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in allowed}).validated()


__all__ = ["SchurConfig"]
