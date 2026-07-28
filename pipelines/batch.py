from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from dct_schur.config import SchurConfig
from .data import embed_payload_file

_IMAGE_SUFFIXES = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


def embed_folder_payload(
    input_folder: str | Path,
    payload_path: str | Path,
    output_folder: str | Path,
    *,
    config: SchurConfig | Mapping[str, Any] | None = None,
) -> Path:
    source = Path(input_folder)
    target = Path(output_folder)
    image_folder = target / "images"
    key_folder = target / "keys"
    image_folder.mkdir(parents=True, exist_ok=True)
    key_folder.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for image_path in sorted(path for path in source.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES):
        output_image = image_folder / f"{image_path.stem}.png"
        output_key = key_folder / f"{image_path.stem}.json"
        result = embed_payload_file(
            image_path, payload_path, output_image, output_key, config=config
        )
        rows.append(
            {
                "input": str(image_path),
                "output": str(result.output_image),
                "key": str(result.key_file),
                "payload_bytes": result.payload_bytes,
                "psnr_db": result.psnr_db,
            }
        )
    manifest = target / "batch_manifest.json"
    manifest.write_text(json.dumps({"items": rows}, indent=2), encoding="utf-8")
    return manifest


__all__ = ["embed_folder_payload"]
