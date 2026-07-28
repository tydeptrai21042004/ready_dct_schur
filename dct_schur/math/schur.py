from __future__ import annotations

from typing import Any
import numpy as np

from dct_schur.constants import COUPLING_POSITIONS, DIAGONAL_POSITIONS, SCALE_POSITIONS


def couplings(coeff: np.ndarray) -> np.ndarray:
    return np.asarray(coeff)[:, COUPLING_POSITIONS[:, 0], COUPLING_POSITIONS[:, 1]]


def write_couplings(coeff: np.ndarray, values: np.ndarray) -> None:
    for index, (u, v) in enumerate(COUPLING_POSITIONS):
        coeff[:, int(u), int(v)] = values[:, index]


def spectral_scale(coeff: np.ndarray) -> np.ndarray:
    values = np.asarray(coeff)[:, SCALE_POSITIONS[:, 0], SCALE_POSITIONS[:, 1]]
    return np.sqrt(np.mean(values * values, axis=1) + 1e-6)


def constructed_matrices(coeff: np.ndarray, lift: float) -> np.ndarray:
    count = np.asarray(coeff).shape[0]
    matrices = np.zeros((count, 4, 4), dtype=np.float64)
    diagonal = np.asarray(coeff)[:, DIAGONAL_POSITIONS[:, 0], DIAGONAL_POSITIONS[:, 1]]
    diagonal = diagonal + float(lift)
    indexes = np.arange(4)
    matrices[:, indexes, indexes] = diagonal
    relation = couplings(coeff)
    upper = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for index, (row, column) in enumerate(upper):
        matrices[:, row, column] = relation[:, index]
    return matrices


def invariant_summary(
    before: np.ndarray,
    after: np.ndarray,
    *,
    lift: float,
    determinant_epsilon: float,
) -> dict[str, Any]:
    matrix_before = constructed_matrices(before, lift)
    matrix_after = constructed_matrices(after, lift)
    eig_before = np.sort_complex(np.linalg.eigvals(matrix_before))
    eig_after = np.sort_complex(np.linalg.eigvals(matrix_after))
    det_before = np.linalg.det(matrix_before)
    det_after = np.linalg.det(matrix_after)
    trace_before = np.trace(matrix_before, axis1=1, axis2=2)
    trace_after = np.trace(matrix_after, axis1=1, axis2=2)
    relative_det = np.abs(det_after - det_before) / np.maximum(
        np.abs(det_before), float(determinant_epsilon)
    )
    return {
        "max_spectrum_error_float": float(np.max(np.abs(eig_after - eig_before))),
        "max_trace_error_float": float(np.max(np.abs(trace_after - trace_before))),
        "max_relative_det_error_float": float(np.max(relative_det)),
        "min_abs_det_float": float(np.min(np.abs(det_after))),
        "all_det_nonzero_float": bool(np.all(np.abs(det_after) > determinant_epsilon)),
    }


__all__ = [
    "couplings", "write_couplings", "spectral_scale",
    "constructed_matrices", "invariant_summary",
]
