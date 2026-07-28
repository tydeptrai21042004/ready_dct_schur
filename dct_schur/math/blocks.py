from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn

from dct_schur.constants import BLOCK_SIZE


def active_shape(image: np.ndarray, minimum_side: int = 512) -> tuple[int, int]:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, received {array.shape}")
    height = (int(array.shape[0]) // BLOCK_SIZE) * BLOCK_SIZE
    width = (int(array.shape[1]) // BLOCK_SIZE) * BLOCK_SIZE
    if height < int(minimum_side) or width < int(minimum_side):
        raise ValueError(
            f"DCT-Schur requires at least {minimum_side}x{minimum_side} active pixels; "
            f"received {array.shape[:2]}"
        )
    return height, width


def opponent_field(rgb: np.ndarray, eta: float = 0.07) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, received {image.shape}")
    return (
        0.299 * image[..., 0]
        + 0.587 * image[..., 1]
        + 0.114 * image[..., 2]
        + float(eta) * (image[..., 0] - 0.5 * image[..., 1] - 0.5 * image[..., 2])
    )


def blockify(field: np.ndarray) -> np.ndarray:
    height, width = np.asarray(field).shape
    if height % BLOCK_SIZE or width % BLOCK_SIZE:
        raise ValueError("Field dimensions must be divisible by eight")
    return (
        np.asarray(field).reshape(height // 8, 8, width // 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 8, 8)
    )


def unblockify(blocks: np.ndarray, height: int, width: int) -> np.ndarray:
    return (
        np.asarray(blocks).reshape(height // 8, width // 8, 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(height, width)
    )


def coefficients(
    image: np.ndarray, eta: float, minimum_side: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    height, width = active_shape(image, minimum_side)
    active = np.asarray(image, dtype=np.uint8)[:height, :width]
    field = opponent_field(active, eta)
    coeff = dctn(blockify(field), type=2, norm="ortho", axes=(-2, -1))
    return field, np.asarray(coeff, dtype=np.float64)


def reconstruct_field(coeff: np.ndarray, height: int, width: int) -> np.ndarray:
    blocks = idctn(np.asarray(coeff), type=2, norm="ortho", axes=(-2, -1))
    return unblockify(blocks, height, width)


def apply_field_delta(base: np.ndarray, delta: np.ndarray, eta: float) -> np.ndarray:
    projection = np.asarray(
        [0.299 + eta, 0.587 - 0.5 * eta, 0.114 - 0.5 * eta], dtype=np.float64
    )
    rgb_direction = projection / float(np.dot(projection, projection))
    output = np.asarray(base, dtype=np.uint8).copy()
    height, width = np.asarray(delta).shape
    updated = (
        output[:height, :width].astype(np.float64)
        + np.asarray(delta, dtype=np.float64)[..., None] * rgb_direction
    )
    output[:height, :width] = np.clip(np.rint(updated), 0, 255).astype(np.uint8)
    return output


__all__ = [
    "active_shape", "opponent_field", "blockify", "unblockify",
    "coefficients", "reconstruct_field", "apply_field_delta",
]
