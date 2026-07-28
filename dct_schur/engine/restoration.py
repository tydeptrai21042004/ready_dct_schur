from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from dct_schur.config import SchurConfig
from .evidence import raw_payload_evidence


def candidate_images(image: np.ndarray, cfg: SchurConfig) -> list[tuple[str, np.ndarray]]:
    base = np.asarray(image, dtype=np.uint8)
    if not cfg.candidate_search_enabled:
        return [("identity", base)]
    pil = Image.fromarray(base)
    output: list[tuple[str, np.ndarray]] = [("identity", base)]
    for factor in cfg.sharpness_factors:
        candidate = ImageEnhance.Sharpness(pil).enhance(float(factor))
        output.append((f"sharpness_{factor:g}", np.asarray(candidate, dtype=np.uint8)))
    for radius, percent in cfg.unsharp_candidates:
        candidate = pil.filter(
            ImageFilter.UnsharpMask(radius=float(radius), percent=int(percent), threshold=0)
        )
        output.append((f"unsharp_r{radius:g}_p{percent}", np.asarray(candidate, dtype=np.uint8)))
    return output


def candidate_evidence_rows(
    image: np.ndarray,
    cfg: SchurConfig,
    reference: np.ndarray,
    *,
    replica_count: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observation_count = 3 * int(replica_count)
    for name, candidate in candidate_images(image, cfg):
        bundle = raw_payload_evidence(
            candidate, cfg, reference, replica_count=replica_count
        )
        agreement = float(np.mean(bundle.consensus))
        evidence_strength = float(
            np.mean(np.abs(bundle.evidence) / max(observation_count, 1))
        )
        mean_confidence = float(np.mean(bundle.confidence))
        mean_witness = float(np.mean(bundle.witness_trust))
        score = (
            cfg.candidate_agreement_weight * agreement
            + cfg.candidate_evidence_weight * evidence_strength
            + cfg.candidate_confidence_weight * mean_confidence
            + cfg.candidate_witness_weight * mean_witness
        )
        rows.append(
            {
                "candidate": name,
                "score": float(score),
                "agreement": agreement,
                "evidence_strength": evidence_strength,
                "mean_confidence": mean_confidence,
                "mean_witness": mean_witness,
                "erasure_fraction": bundle.erasure_fraction,
                "evidence": bundle.evidence,
                "confidence": bundle.confidence,
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows


__all__ = ["candidate_images", "candidate_evidence_rows"]
