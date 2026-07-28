"""Common benchmark layer for the DCT-Schur proposal and published baselines."""

from .registry import (
    DCT_SCHUR_ID,
    all_method_specs,
    get_method_spec,
    resolve_methods,
)
from .runner import evaluate_method_trial, run_benchmark_matrix

__all__ = [
    "DCT_SCHUR_ID",
    "all_method_specs",
    "get_method_spec",
    "resolve_methods",
    "evaluate_method_trial",
    "run_benchmark_matrix",
]
