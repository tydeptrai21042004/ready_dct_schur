from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from baselines import embed_baseline, extract_baseline
from dct_schur import SchurConfig, embed_plane, extract_plane

from .registry import DCT_SCHUR_ID
from .types import BaselineParameters, MethodSpec


class MethodAdapter:
    """A narrow common interface; proposal and baseline internals remain isolated."""

    def __init__(
        self,
        spec: MethodSpec,
        *,
        schur_config: SchurConfig | None = None,
        baseline_parameters: BaselineParameters | None = None,
    ) -> None:
        self.spec = spec
        self.schur_config = schur_config or SchurConfig()
        self.baseline_parameters = baseline_parameters or BaselineParameters()

    def configuration_summary(self, seed: int) -> dict[str, Any]:
        if self.spec.method_id == DCT_SCHUR_ID:
            config = replace(self.schur_config, seed=int(seed)).validated()
            return config.to_dict()
        return {
            "repeat": self.baseline_parameters.repeat,
            "step": self.baseline_parameters.step,
            "method_params": dict(self.baseline_parameters.method_params),
        }

    def embed(
        self,
        host: np.ndarray,
        payload: np.ndarray,
        *,
        seed: int,
    ) -> tuple[np.ndarray, Any, dict[str, Any]]:
        if self.spec.method_id == DCT_SCHUR_ID:
            config = replace(self.schur_config, seed=int(seed)).validated()
            marked, key, metadata = embed_plane(
                host,
                payload,
                config=config,
                return_metadata=True,
            )
            return np.asarray(marked, dtype=np.uint8), key, dict(metadata or {})
        params = self.baseline_parameters
        marked, key = embed_baseline(
            self.spec.method_id,
            host,
            payload,
            seed=int(seed),
            repeat=params.repeat,
            step=params.step,
            method_params=dict(params.method_params),
        )
        return np.asarray(marked, dtype=np.uint8), key, {}

    def extract(
        self,
        image: np.ndarray,
        key: Any,
        *,
        original_host: np.ndarray | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.spec.method_id == DCT_SCHUR_ID:
            recovered, metadata = extract_plane(image, key, return_metadata=True)
            return np.asarray(recovered, dtype=np.uint8), dict(metadata or {})
        recovered = extract_baseline(
            image,
            key,
            original_host=original_host if self.spec.requires_original_host else None,
        )
        return np.asarray(recovered, dtype=np.uint8), {}


__all__ = ["MethodAdapter"]
