from __future__ import annotations

from typing import Any
import numpy as np

from dct_schur.config import SchurConfig
from dct_schur.constants import METHOD_ID, PAYLOAD_BITS
from dct_schur.key import SchurKey
from dct_schur.math.arnold import arnold_inverse
from dct_schur.math.blocks import active_shape, coefficients
from dct_schur.math.coset import unpack_binary_mask
from dct_schur.math.layout import effective_step, payload_block_layout
from dct_schur.math.schur import constructed_matrices
from .restoration import candidate_evidence_rows


def _validate_image(image: np.ndarray, key: SchurKey, cfg: SchurConfig) -> tuple[np.ndarray, tuple[int, int]]:
    array = np.asarray(image, dtype=np.uint8)
    if tuple(array.shape) != tuple(key.host_shape):
        raise ValueError(f"Image shape {array.shape} does not match key shape {key.host_shape}")
    active = active_shape(array, cfg.minimum_host_side)
    if key.active_shape is not None and tuple(active) != tuple(key.active_shape):
        raise ValueError(f"Active shape {active} does not match key active shape {key.active_shape}")
    return array, active


def extract_evidence_candidates(
    possibly_attacked_rgb: np.ndarray,
    key: SchurKey,
    *,
    return_metadata: bool = False,
):
    cfg = SchurConfig.from_mapping(key.config)
    image, active = _validate_image(possibly_attacked_rgb, key, cfg)
    replica_count = int(key.replica_count or 1)
    reference = np.asarray(key.spectral_reference, dtype=np.float64)
    expected = replica_count * PAYLOAD_BITS
    if reference.size != expected:
        raise ValueError(f"Key has {reference.size} spectral values; expected {expected}")

    rows = candidate_evidence_rows(image, cfg, reference, replica_count=replica_count)
    flips = unpack_binary_mask(key.payload_coset_flip_b64, PAYLOAD_BITS)
    flip_sign = np.where(flips > 0, -1.0, 1.0)
    candidates: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        logical = np.asarray(row["evidence"], dtype=np.float64) * flip_sign
        evidence_map = arnold_inverse(
            logical.reshape(key.payload_shape), cfg.arnold_iterations, key.arnold_period
        )
        candidates.append(
            {
                "rank": rank,
                "candidate": str(row["candidate"]),
                "score": float(row["score"]),
                "agreement": float(row["agreement"]),
                "evidence_strength": float(row["evidence_strength"]),
                "mean_confidence": float(row["mean_confidence"]),
                "mean_witness": float(row["mean_witness"]),
                "erasure_fraction": float(row["erasure_fraction"]),
                "evidence": evidence_map,
            }
        )

    metadata = {
        "method_id": METHOD_ID,
        "inference_path": "witness_gated_robust_schur_list_evidence",
        "candidate_count": len(candidates),
        "candidate_scores": [
            {key_name: value for key_name, value in candidate.items() if key_name != "evidence"}
            for candidate in candidates
        ],
        "replica_count": replica_count,
        "observations_per_payload_bit": 3 * replica_count,
        "effective_replica_step": effective_step(
            cfg.step, replica_count, cfg.replica_step_exponent
        ),
        "active_shape": list(active),
        "witness_gating_enabled": cfg.witness_gating_enabled,
        "robust_consensus_enabled": cfg.robust_consensus_enabled,
        "blindness_class": key.blindness_class,
    }
    return (candidates, metadata) if return_metadata else candidates


def extract_evidence(
    possibly_attacked_rgb: np.ndarray,
    key: SchurKey,
    *,
    return_metadata: bool = False,
):
    candidates, metadata = extract_evidence_candidates(
        possibly_attacked_rgb, key, return_metadata=True
    )
    selected = candidates[0]
    output_metadata = dict(metadata)
    output_metadata.update(
        {
            "selected_candidate": selected["candidate"],
            "selected_candidate_rank": selected["rank"],
            "copy_agreement": selected["agreement"],
            "mean_witness": selected["mean_witness"],
            "erasure_fraction": selected["erasure_fraction"],
        }
    )
    evidence = np.asarray(selected["evidence"], dtype=np.float64)

    cfg = SchurConfig.from_mapping(key.config)
    image = np.asarray(possibly_attacked_rgb, dtype=np.uint8)
    _, coeff = coefficients(image, cfg.eta, cfg.minimum_host_side)
    layout = payload_block_layout(
        coeff.shape[0], seed=cfg.seed, replicas_enabled=key.replica_count > 1,
        max_replicas=key.replica_count,
    )
    matrices = constructed_matrices(coeff[layout.reshape(-1)], cfg.schur_lift)
    determinants = np.abs(np.linalg.det(matrices))
    output_metadata.update(
        {
            "det_nonzero": bool(np.all(determinants > cfg.determinant_epsilon)),
            "min_abs_det": float(np.min(determinants)),
            "median_abs_det": float(np.median(determinants)),
        }
    )
    return (evidence, output_metadata) if return_metadata else evidence


def _icm_map(evidence: np.ndarray, lam: float, iterations: int) -> np.ndarray:
    data = np.asarray(evidence, dtype=np.float64)
    state = np.where(data >= 0.0, 1.0, -1.0)
    if lam <= 0 or iterations <= 0:
        return (state > 0).astype(np.uint8) * 255
    for _ in range(int(iterations)):
        neighbour = np.zeros_like(state)
        neighbour[1:, :] += state[:-1, :]
        neighbour[:-1, :] += state[1:, :]
        neighbour[:, 1:] += state[:, :-1]
        neighbour[:, :-1] += state[:, 1:]
        updated = np.where(data + float(lam) * neighbour >= 0.0, 1.0, -1.0)
        if np.array_equal(updated, state):
            break
        state = updated
    return (state > 0).astype(np.uint8) * 255


def extract_plane(
    possibly_attacked_rgb: np.ndarray,
    key: SchurKey,
    *,
    return_metadata: bool = False,
):
    evidence, metadata = extract_evidence(
        possibly_attacked_rgb, key, return_metadata=True
    )
    cfg = SchurConfig.from_mapping(key.config)
    recovered = _icm_map(evidence, cfg.map_lambda, cfg.map_iters)
    return (recovered, metadata) if return_metadata else recovered


__all__ = ["extract_evidence_candidates", "extract_evidence", "extract_plane"]
