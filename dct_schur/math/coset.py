from __future__ import annotations

import base64
from typing import Any
import numpy as np

from dct_schur.constants import COUPLING_BASIS, PAYLOAD_BITS
from .qim import nearest_parity_target
from .schur import couplings


def pack_binary_mask(mask: np.ndarray) -> str:
    bits = np.asarray(mask, dtype=np.uint8).reshape(-1) & 1
    return base64.b64encode(np.packbits(bits, bitorder="little").tobytes()).decode("ascii")


def unpack_binary_mask(value: str, count: int) -> np.ndarray:
    if not value:
        return np.zeros(int(count), dtype=np.uint8)
    raw = base64.b64decode(value.encode("ascii"), validate=True)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
    if bits.size < int(count):
        raise ValueError("payload coset mask is shorter than the payload")
    return bits[: int(count)].astype(np.uint8, copy=True)


def optimize_payload_coset(
    coeff: np.ndarray,
    payload: np.ndarray,
    permutations_by_replica: tuple[tuple[np.ndarray, ...], ...],
    *,
    step: float,
    block_layout: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    logical = np.asarray(payload, dtype=np.uint8).reshape(-1) & 1
    if logical.size != PAYLOAD_BITS:
        raise ValueError(f"payload must contain {PAYLOAD_BITS} bits")
    layout = np.asarray(block_layout, dtype=np.int32)
    if layout.ndim != 2 or layout.shape[1] != PAYLOAD_BITS:
        raise ValueError("block_layout must have shape (replicas,4096)")
    if len(permutations_by_replica) != layout.shape[0]:
        raise ValueError("one three-channel permutation tuple is required per replica")

    all_couplings = couplings(coeff)
    energy_for_label = np.zeros((2, logical.size), dtype=np.float64)
    for replica, indexes in enumerate(layout):
        carriers = all_couplings[indexes] @ COUPLING_BASIS.T
        for channel, permutation in enumerate(permutations_by_replica[replica]):
            inverse = np.argsort(permutation)
            ordered_carrier = carriers[inverse, channel]
            for label in (0, 1):
                target = nearest_parity_target(
                    ordered_carrier,
                    step,
                    np.full(logical.size, label, dtype=np.uint8),
                )
                energy_for_label[label] += (target - ordered_carrier) ** 2

    index = np.arange(logical.size)
    original_energy = energy_for_label[logical, index]
    complement_energy = energy_for_label[1 - logical, index]
    flips = (complement_energy < original_energy).astype(np.uint8)
    optimized = np.minimum(original_energy, complement_energy)
    total_original = float(np.sum(original_energy))
    total_optimized = float(np.sum(optimized))
    return flips, {
        "coset_projection_energy_before": total_original,
        "coset_projection_energy_after": total_optimized,
        "coset_projection_energy_ratio": total_optimized / max(total_original, np.finfo(float).tiny),
        "coset_flip_fraction": float(np.mean(flips)),
        "coset_observations_per_bit": float(3 * layout.shape[0]),
    }


__all__ = ["pack_binary_mask", "unpack_binary_mask", "optimize_payload_coset"]
