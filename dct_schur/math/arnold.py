from __future__ import annotations

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=32)
def arnold_period(size: int, max_iter: int = 8192) -> int:
    common = {16: 12, 32: 24, 64: 48, 128: 96, 256: 192}
    if int(size) in common:
        return common[int(size)]
    n = int(size)
    initial = np.arange(n * n, dtype=np.int32).reshape(n, n)
    current = initial.copy()
    x, y = np.indices((n, n))
    rr = (x + y) % n
    cc = (x + 2 * y) % n
    for period in range(1, int(max_iter) + 1):
        updated = np.empty_like(current)
        updated[rr, cc] = current
        current = updated
        if np.array_equal(current, initial):
            return period
    raise RuntimeError(f"Arnold period not found for n={n}")


def arnold_transform(array: np.ndarray, iterations: int) -> np.ndarray:
    source = np.asarray(array).copy()
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("Arnold transform requires a square array")
    n = source.shape[0]
    x, y = np.indices((n, n))
    rr = (x + y) % n
    cc = (x + 2 * y) % n
    for _ in range(int(iterations)):
        updated = np.empty_like(source)
        updated[rr, cc] = source
        source = updated
    return source


def arnold_inverse(array: np.ndarray, iterations: int, period: int) -> np.ndarray:
    inverse_iterations = (int(period) - int(iterations) % int(period)) % int(period)
    return arnold_transform(array, inverse_iterations)


__all__ = ["arnold_period", "arnold_transform", "arnold_inverse"]
