from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter

from dct_schur.config import SchurConfig
from dct_schur.constants import COUPLING_BASIS, PAYLOAD_BITS, PAYLOAD_SIDE
from dct_schur.math.blocks import coefficients
from dct_schur.math.layout import effective_step, payload_block_layout, payload_permutations
from dct_schur.math.schur import couplings, spectral_scale


@dataclass(frozen=True)
class EvidenceBundle:
    evidence: np.ndarray
    confidence: np.ndarray
    votes: np.ndarray
    witness_trust: np.ndarray
    consensus: np.ndarray
    selected_scales: np.ndarray
    erasure_fraction: float


def _witness_trust(
    current_scale: np.ndarray,
    reference_scale: np.ndarray,
    cfg: SchurConfig,
) -> tuple[np.ndarray, np.ndarray]:
    raw_ratio = np.asarray(current_scale, dtype=np.float64) / np.maximum(
        np.asarray(reference_scale, dtype=np.float64), 1e-6
    )
    log_ratio = np.log(np.maximum(raw_ratio, 1e-8))
    global_shift = float(np.median(log_ratio))
    local_drift = log_ratio - global_shift
    if not cfg.witness_gating_enabled:
        return np.ones_like(raw_ratio), raw_ratio
    gaussian = np.exp(-0.5 * (local_drift / cfg.witness_log_sigma) ** 2)
    trust = cfg.witness_floor + (1.0 - cfg.witness_floor) * gaussian
    return np.clip(trust, cfg.witness_floor, 1.0), raw_ratio


def _robust_consensus(
    votes: np.ndarray,
    witness_votes: np.ndarray,
    cfg: SchurConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    matrix = np.asarray(votes, dtype=np.float64)
    signs = np.sign(matrix)
    consensus = np.abs(np.mean(signs, axis=0))
    mean_witness = np.mean(np.asarray(witness_votes, dtype=np.float64), axis=0)

    if cfg.robust_consensus_enabled:
        median = np.median(matrix, axis=0)
        magnitude = np.median(np.abs(matrix), axis=0)
        limit = np.maximum(cfg.consensus_clip * magnitude, 1e-6)
        clipped_mean = np.mean(np.clip(matrix, -limit, limit), axis=0)
        weight_sum = cfg.consensus_median_weight + cfg.consensus_mean_weight
        robust_location = (
            cfg.consensus_median_weight * median
            + cfg.consensus_mean_weight * clipped_mean
        ) / weight_sum
        evidence = robust_location * matrix.shape[0]
    else:
        evidence = np.sum(matrix, axis=0)

    reliability = consensus * mean_witness
    low = reliability < cfg.erasure_threshold
    if cfg.erasure_threshold > 0:
        attenuation = np.minimum(1.0, reliability / cfg.erasure_threshold)
        evidence = evidence * attenuation
    return evidence, consensus, mean_witness, float(np.mean(low))


def raw_payload_evidence(
    image: np.ndarray,
    cfg: SchurConfig,
    reference: np.ndarray,
    *,
    replica_count: int = 1,
) -> EvidenceBundle:
    _, coeff = coefficients(image, cfg.eta, cfg.minimum_host_side)
    layout = payload_block_layout(
        coeff.shape[0],
        seed=cfg.seed,
        replicas_enabled=int(replica_count) > 1,
        max_replicas=int(replica_count),
    )
    if layout.shape[0] != int(replica_count):
        raise ValueError(
            f"Image supports {layout.shape[0]} replicas but key requires {replica_count}"
        )
    reference_matrix = np.asarray(reference, dtype=np.float64).reshape(replica_count, PAYLOAD_BITS)
    all_relations = couplings(coeff)
    all_scales = spectral_scale(coeff)
    step = effective_step(cfg.step, replica_count, cfg.replica_step_exponent)

    votes: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    witness_rows: list[np.ndarray] = []
    selected_scales: list[np.ndarray] = []
    for replica, indexes in enumerate(layout):
        relation = all_relations[indexes]
        current_scale = all_scales[indexes]
        selected_scales.append(current_scale)
        witness, raw_ratio = _witness_trust(current_scale, reference_matrix[replica], cfg)

        gain_ratio: np.ndarray | None = None
        if cfg.gain_normalization_enabled:
            if (
                replica_count == 1
                and coeff.shape[0] == PAYLOAD_BITS
                and np.array_equal(indexes, np.arange(PAYLOAD_BITS))
            ):
                gain_ratio = median_filter(
                    raw_ratio.reshape(PAYLOAD_SIDE, PAYLOAD_SIDE), size=3, mode="nearest"
                ).reshape(-1)
            else:
                median_ratio = float(np.median(raw_ratio))
                gain_ratio = 0.5 * raw_ratio + 0.5 * median_ratio
            gain_ratio = np.clip(gain_ratio, cfg.gain_clip[0], cfg.gain_clip[1])

        permutations = payload_permutations(cfg.seed, PAYLOAD_BITS, replica=replica)
        for channel in range(3):
            carrier = relation @ COUPLING_BASIS[channel]
            if gain_ratio is not None:
                carrier = carrier / np.power(gain_ratio, cfg.gain_gamma)
            normalized = carrier / step
            quantized = np.rint(normalized).astype(np.int64)
            bits = (quantized & 1).astype(np.uint8)
            confidence = np.clip(1.0 - 2.0 * np.abs(normalized - quantized), 0.0, 1.0)
            magnitude = cfg.confidence_floor + cfg.confidence_scale * np.power(
                confidence, cfg.confidence_power
            )
            signed = np.where(bits > 0, 1.0, -1.0) * magnitude * witness

            payload_vote = np.zeros(PAYLOAD_BITS, dtype=np.float64)
            payload_confidence = np.zeros(PAYLOAD_BITS, dtype=np.float64)
            payload_witness = np.zeros(PAYLOAD_BITS, dtype=np.float64)
            payload_vote[permutations[channel]] = signed
            payload_confidence[permutations[channel]] = confidence
            payload_witness[permutations[channel]] = witness
            votes.append(payload_vote)
            confidences.append(payload_confidence)
            witness_rows.append(payload_witness)

    vote_matrix = np.asarray(votes, dtype=np.float64)
    confidence_matrix = np.asarray(confidences, dtype=np.float64)
    witness_matrix = np.asarray(witness_rows, dtype=np.float64)
    evidence, consensus, mean_witness, erasure_fraction = _robust_consensus(
        vote_matrix, witness_matrix, cfg
    )
    return EvidenceBundle(
        evidence=evidence,
        confidence=np.mean(confidence_matrix, axis=0),
        votes=vote_matrix,
        witness_trust=mean_witness,
        consensus=consensus,
        selected_scales=np.asarray(selected_scales, dtype=np.float64),
        erasure_fraction=erasure_fraction,
    )


__all__ = ["EvidenceBundle", "raw_payload_evidence"]
