from __future__ import annotations

import numpy as np

METHOD_ID = "dct_schur"
SCIENTIFIC_NAME = "Invariant-Relational DCT-Schur Watermarking"
SHORT_NAME = "IR-DSW"
BLOCK_SIZE = 8
PAYLOAD_SIDE = 64
PAYLOAD_BITS = PAYLOAD_SIDE * PAYLOAD_SIDE

# Six DCT coefficients become the strict-upper coordinates
# (t12,t13,t14,t23,t24,t34) of a virtual 4x4 triangular Schur form.
COUPLING_POSITIONS = np.asarray(
    [(0, 1), (1, 0), (0, 2), (1, 1), (2, 0), (1, 2)], dtype=np.int32
)

# Disjoint coefficients define the local spectral witness. They are never
# changed by the embedding projection.
DIAGONAL_POSITIONS = np.asarray(
    [(2, 1), (0, 3), (3, 0), (2, 2)], dtype=np.int32
)
SCALE_POSITIONS = np.asarray(
    [(2, 1), (0, 3), (3, 0), (2, 2), (1, 3), (3, 1)], dtype=np.int32
)

# Orthonormal relational directions in the six-dimensional strict-upper space.
COUPLING_BASIS = np.asarray(
    [
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, -1.0, -1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, -2.0, -2.0],
    ],
    dtype=np.float64,
)
COUPLING_BASIS /= np.linalg.norm(COUPLING_BASIS, axis=1, keepdims=True)

__all__ = [
    "METHOD_ID", "SCIENTIFIC_NAME", "SHORT_NAME", "BLOCK_SIZE",
    "PAYLOAD_SIDE", "PAYLOAD_BITS", "COUPLING_POSITIONS",
    "DIAGONAL_POSITIONS", "SCALE_POSITIONS", "COUPLING_BASIS",
]
