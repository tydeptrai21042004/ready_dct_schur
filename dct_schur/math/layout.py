from __future__ import annotations

import numpy as np

from dct_schur.constants import PAYLOAD_BITS


def payload_permutations(seed: int, count: int = PAYLOAD_BITS, replica: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(int(seed) + 104729 * int(replica))
    return tuple(rng.permutation(count).astype(np.int32) for _ in range(3))


def effective_step(base_step: float, replica_count: int, exponent: float) -> float:
    return float(base_step) / (max(int(replica_count), 1) ** float(exponent))


def payload_block_layout(
    block_count: int,
    *,
    seed: int,
    replicas_enabled: bool,
    max_replicas: int,
) -> np.ndarray:
    count = int(block_count)
    if count < PAYLOAD_BITS:
        raise ValueError(f"At least {PAYLOAD_BITS} DCT blocks are required; received {count}")
    replica_count = 1
    if replicas_enabled:
        replica_count = min(int(max_replicas), count // PAYLOAD_BITS)
    if count == PAYLOAD_BITS and replica_count == 1:
        return np.arange(PAYLOAD_BITS, dtype=np.int32).reshape(1, PAYLOAD_BITS)
    rng = np.random.default_rng(int(seed) ^ 0x5C4A7E11)
    chosen = rng.permutation(count)[: replica_count * PAYLOAD_BITS]
    return np.asarray(chosen, dtype=np.int32).reshape(replica_count, PAYLOAD_BITS)


__all__ = ["payload_permutations", "effective_step", "payload_block_layout"]
