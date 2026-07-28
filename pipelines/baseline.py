from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baselines import embed_baseline, extract_baseline, get_baseline_spec
from benchmarking.runner import prepare_payload
from dct_schur.io import load_binary_plane, load_rgb, save_binary_plane, save_rgb
from evaluation.metrics import nc, psnr


@dataclass(frozen=True)
class BaselineEmbedResult:
    method_id: str
    output_image: Path
    key_file: Path
    payload_bits: int
    psnr_db: float
    clean_nc: float


def embed_baseline_image(
    method_id: str,
    host_path: str | Path,
    payload_path: str | Path,
    output_image_path: str | Path,
    key_path: str | Path,
    *,
    seed: int = 2026,
    repeat: int | str = 1,
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
) -> BaselineEmbedResult:
    spec = get_baseline_spec(method_id)
    host = load_rgb(host_path)
    source_payload = load_binary_plane(payload_path, side=64)
    payload_size = 64 if spec.common_4096_payload else 16
    payload = prepare_payload(source_payload, payload_size)
    marked, key = embed_baseline(
        spec.canonical_id,
        host,
        payload,
        seed=seed,
        repeat=repeat,
        step=step,
        method_params=method_params,
    )
    recovered = extract_baseline(
        marked,
        key,
        original_host=host if spec.requires_original_host else None,
    )
    output = save_rgb(output_image_path, marked)
    key_file = Path(key_path)
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_bytes(pickle.dumps({"key": key, "original_host": host if spec.requires_original_host else None}))
    return BaselineEmbedResult(
        spec.canonical_id,
        output,
        key_file,
        int(payload.size),
        psnr(host, marked),
        nc(payload, recovered),
    )


def extract_baseline_image(
    image_path: str | Path,
    key_path: str | Path,
    output_payload_path: str | Path,
) -> Path:
    bundle = pickle.loads(Path(key_path).read_bytes())
    key = bundle["key"]
    image = load_rgb(image_path)
    recovered = extract_baseline(image, key, original_host=bundle.get("original_host"))
    return save_binary_plane(output_payload_path, recovered)


__all__ = ["BaselineEmbedResult", "embed_baseline_image", "extract_baseline_image"]
