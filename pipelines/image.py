from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dct_schur import SchurConfig, SchurKey, embed_plane, extract_plane
from dct_schur.io import load_binary_plane, load_rgb, save_binary_plane, save_rgb
from evaluation.metrics import nc, psnr


@dataclass(frozen=True)
class ImagePipelineResult:
    output_image: Path
    key_file: Path
    psnr_db: float
    clean_nc: float


def embed_image(
    host_path: str | Path,
    payload_image_path: str | Path,
    output_image_path: str | Path,
    key_path: str | Path,
    *,
    config: SchurConfig | Mapping[str, Any] | None = None,
) -> ImagePipelineResult:
    host = load_rgb(host_path)
    payload = load_binary_plane(payload_image_path)
    marked, key = embed_plane(host, payload, config=config)
    recovered = extract_plane(marked, key)
    output = save_rgb(output_image_path, marked)
    key_file = key.save(key_path)
    return ImagePipelineResult(output, key_file, psnr(host, marked), nc(payload, recovered))


def extract_image(
    image_path: str | Path,
    key_path: str | Path,
    output_payload_path: str | Path,
) -> Path:
    image = load_rgb(image_path)
    key = SchurKey.load(key_path)
    recovered = extract_plane(image, key)
    return save_binary_plane(output_payload_path, recovered)


__all__ = ["ImagePipelineResult", "embed_image", "extract_image"]
