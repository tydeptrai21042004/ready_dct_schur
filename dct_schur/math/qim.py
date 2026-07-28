from __future__ import annotations

from typing import Any
import numpy as np

from dct_schur.constants import COUPLING_BASIS
from .schur import couplings, write_couplings


def nearest_parity_target(carrier: np.ndarray, step: float, bits: np.ndarray) -> np.ndarray:
    normalized = np.asarray(carrier, dtype=np.float64) / float(step)
    quantized = np.rint(normalized).astype(np.int64)
    desired = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    wrong = (quantized & 1) != desired
    left = quantized - 1
    right = quantized + 1
    alternate = np.where(
        np.abs(normalized - left) <= np.abs(normalized - right), left, right
    )
    quantized = np.where(wrong, alternate, quantized)
    return quantized.astype(np.float64) * float(step)


def project_constraints(
    coeff: np.ndarray,
    bits_by_carrier: np.ndarray,
    *,
    step: float,
) -> dict[str, Any]:
    relation_before = couplings(coeff).copy()
    carriers_before = relation_before @ COUPLING_BASIS.T
    targets = np.empty_like(carriers_before)
    for channel in range(3):
        targets[:, channel] = nearest_parity_target(
            carriers_before[:, channel], step, bits_by_carrier[channel]
        )
    relation_after = relation_before + (targets - carriers_before) @ COUPLING_BASIS
    write_couplings(coeff, relation_after)
    achieved = relation_after @ COUPLING_BASIS.T
    residual = achieved - targets
    return {
        "max_constraint_residual": float(np.max(np.abs(residual))),
        "mean_constraint_residual": float(np.mean(np.abs(residual))),
        "projection_energy": float(np.sum((relation_after - relation_before) ** 2)),
        "mean_projection_energy_per_block": float(
            np.mean(np.sum((relation_after - relation_before) ** 2, axis=1))
        ),
    }


def project_replicas(
    coeff: np.ndarray,
    block_layout: np.ndarray,
    bits_by_replica: tuple[np.ndarray, ...],
    *,
    step: float,
) -> dict[str, Any]:
    layout = np.asarray(block_layout, dtype=np.int32)
    if len(bits_by_replica) != layout.shape[0]:
        raise ValueError("bits_by_replica must match the replica layout")
    rows: list[dict[str, Any]] = []
    for replica, indexes in enumerate(layout):
        selected = np.asarray(coeff[indexes], dtype=np.float64).copy()
        stats = project_constraints(selected, bits_by_replica[replica], step=step)
        coeff[indexes] = selected
        rows.append({"replica": int(replica), **stats})
    return {
        "replica_count": int(layout.shape[0]),
        "max_constraint_residual": float(max(row["max_constraint_residual"] for row in rows)),
        "mean_constraint_residual": float(np.mean([row["mean_constraint_residual"] for row in rows])),
        "projection_energy": float(sum(row["projection_energy"] for row in rows)),
        "mean_projection_energy_per_used_block": float(
            np.mean([row["mean_projection_energy_per_block"] for row in rows])
        ),
        "replica_projection_stats": rows,
    }


__all__ = ["nearest_parity_target", "project_constraints", "project_replicas"]
