from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from dct_schur.config import SchurConfig
from dct_schur.constants import (
    COUPLING_BASIS, METHOD_ID, PAYLOAD_BITS, PAYLOAD_SIDE, SCIENTIFIC_NAME,
)
from dct_schur.key import SchurKey
from dct_schur.math.arnold import arnold_period, arnold_transform
from dct_schur.math.blocks import active_shape, apply_field_delta, coefficients, reconstruct_field
from dct_schur.math.coset import optimize_payload_coset, pack_binary_mask
from dct_schur.math.layout import effective_step, payload_block_layout, payload_permutations
from dct_schur.math.qim import project_replicas
from dct_schur.math.schur import couplings, invariant_summary, spectral_scale


def embed_plane(
    host_rgb: np.ndarray,
    payload_plane: np.ndarray,
    *,
    config: SchurConfig | Mapping[str, Any] | None = None,
    return_metadata: bool = False,
):
    """Embed one 64x64 bit plane through invariant-preserving Schur projection."""
    cfg = SchurConfig.from_mapping(config)
    host = np.asarray(host_rgb, dtype=np.uint8)
    payload_image = (np.asarray(payload_plane, dtype=np.uint8) > 0).astype(np.uint8)
    if payload_image.shape != (PAYLOAD_SIDE, PAYLOAD_SIDE):
        raise ValueError(
            f"DCT-Schur requires a {PAYLOAD_SIDE}x{PAYLOAD_SIDE} payload plane; "
            f"received {payload_image.shape}"
        )
    active_height, active_width = active_shape(host, cfg.minimum_host_side)
    logical_payload = arnold_transform(payload_image, cfg.arnold_iterations).reshape(-1)

    _, initial_coeff = coefficients(host, cfg.eta, cfg.minimum_host_side)
    layout = payload_block_layout(
        initial_coeff.shape[0],
        seed=cfg.seed,
        replicas_enabled=cfg.resolution_adaptive_replicas_enabled,
        max_replicas=cfg.max_payload_replicas,
    )
    replica_count = int(layout.shape[0])
    step = effective_step(cfg.step, replica_count, cfg.replica_step_exponent)
    permutations = tuple(
        payload_permutations(cfg.seed, PAYLOAD_BITS, replica=replica)
        for replica in range(replica_count)
    )

    flips = np.zeros(PAYLOAD_BITS, dtype=np.uint8)
    coset_stats = {
        "coset_projection_energy_before": 0.0,
        "coset_projection_energy_after": 0.0,
        "coset_projection_energy_ratio": 1.0,
        "coset_flip_fraction": 0.0,
        "coset_observations_per_bit": float(3 * replica_count),
    }
    if cfg.payload_coset_optimization_enabled:
        flips, coset_stats = optimize_payload_coset(
            initial_coeff,
            logical_payload,
            permutations,
            step=step,
            block_layout=layout,
        )
    physical_payload = logical_payload ^ flips
    bits_by_replica = tuple(
        np.vstack([physical_payload[permutation] for permutation in replica_permutations])
        for replica_permutations in permutations
    )

    current = host.copy()
    closure_history: list[dict[str, Any]] = []
    invariant_stats: dict[str, Any] = {}
    used_indexes = layout.reshape(-1)
    for iteration in range(cfg.closure_rounds):
        field, coeff = coefficients(current, cfg.eta, cfg.minimum_host_side)
        before_used = coeff[used_indexes].copy()
        projection_stats = project_replicas(coeff, layout, bits_by_replica, step=step)
        invariant_stats = invariant_summary(
            before_used,
            coeff[used_indexes],
            lift=cfg.schur_lift,
            determinant_epsilon=cfg.determinant_epsilon,
        )
        reconstructed = reconstruct_field(coeff, field.shape[0], field.shape[1])
        current = apply_field_delta(current, reconstructed - field, cfg.eta)

        _, rounded_coeff = coefficients(current, cfg.eta, cfg.minimum_host_side)
        rounded_relations = couplings(rounded_coeff)
        replica_accuracy: list[list[float]] = []
        all_exact = True
        for replica, indexes in enumerate(layout):
            selected = rounded_relations[indexes]
            copy_accuracy: list[float] = []
            for channel in range(3):
                quantized = np.rint((selected @ COUPLING_BASIS[channel]) / step).astype(np.int64)
                decoded = (quantized & 1).astype(np.uint8)
                correct = decoded == bits_by_replica[replica][channel]
                copy_accuracy.append(float(np.mean(correct)))
                all_exact = all_exact and bool(np.all(correct))
            replica_accuracy.append(copy_accuracy)
        closure_history.append(
            {
                "iteration": iteration + 1,
                "replica_copy_accuracy": replica_accuracy,
                "all_copies_exact": all_exact,
                **projection_stats,
            }
        )
        if all_exact:
            break

    _, final_coeff = coefficients(current, cfg.eta, cfg.minimum_host_side)
    reference = spectral_scale(final_coeff)[layout].reshape(-1)
    key = SchurKey(
        host_shape=tuple(int(value) for value in host.shape),
        payload_shape=(PAYLOAD_SIDE, PAYLOAD_SIDE),
        config=cfg.to_dict(),
        spectral_reference=[float(value) for value in reference],
        arnold_period=arnold_period(PAYLOAD_SIDE),
        payload_coset_flip_b64=pack_binary_mask(flips),
        active_shape=(active_height, active_width),
        replica_count=replica_count,
    )
    metadata = {
        "method_id": METHOD_ID,
        "scientific_name": SCIENTIFIC_NAME,
        "core_philosophy": "preserve spectral identity; inscribe relational couplings",
        "identity_component": "unchanged virtual Schur diagonal",
        "relational_component": "three orthogonal strict-upper coupling carriers",
        "embedding_law": "n*=n+H^T(t-Hn)",
        "minimum_frobenius_projection": True,
        "spectrum_preserved_float": invariant_stats.get("max_spectrum_error_float", 1.0) < 1e-8,
        "trace_preserved_float": invariant_stats.get("max_trace_error_float", 1.0) < 1e-8,
        "determinant_preserved_float": invariant_stats.get("max_relative_det_error_float", 1.0) < 1e-8,
        "replica_count": replica_count,
        "observations_per_payload_bit": 3 * replica_count,
        "base_step": float(cfg.step),
        "effective_replica_step": float(step),
        "active_shape": [active_height, active_width],
        "blindness_class": key.blindness_class,
        "closure_history": closure_history,
        **coset_stats,
        **invariant_stats,
    }
    return (current, key, metadata) if return_metadata else (current, key)


__all__ = ["embed_plane"]
