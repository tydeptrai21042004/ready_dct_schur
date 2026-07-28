from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from dct_schur.config import SchurConfig
from dct_schur.io import load_rgb, save_rgb
from dct_schur.provenance import MediaType, embed_sequence, verify_sequence
from dct_schur.provenance.keys import load_private_key, load_public_key
from dct_schur.transport import DataKey

_IMAGE_SUFFIXES = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


def embed_document_pages(
    page_folder: str | Path,
    output_folder: str | Path,
    private_key_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    config: SchurConfig | Mapping[str, Any] | None = None,
) -> Path:
    source = Path(page_folder)
    target = Path(output_folder)
    marked_folder = target / "pages"
    marked_folder.mkdir(parents=True, exist_ok=True)
    page_paths = sorted(
        path for path in source.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not page_paths:
        raise ValueError("page folder contains no supported images")
    frames = [load_rgb(path) for path in page_paths]
    manifest = Path(manifest_path).read_bytes() if manifest_path is not None else None
    marked, keys, records = embed_sequence(
        frames,
        private_key=load_private_key(private_key_path),
        manifest_bytes=manifest,
        media_type=MediaType.DOCUMENT_PAGE,
        config=config,
    )
    outputs: list[str] = []
    for index, (source_path, image) in enumerate(zip(page_paths, marked), start=1):
        output = marked_folder / f"page_{index:04d}_{source_path.stem}.png"
        save_rgb(output, image)
        outputs.append(str(output))
    bundle = {
        "source_pages": [str(path) for path in page_paths],
        "marked_pages": outputs,
        "keys": [key.to_dict() for key in keys],
        "records": [record.summary() for record in records],
    }
    bundle_path = target / "document_provenance_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle_path


def verify_document_pages(
    bundle_path: str | Path,
    public_key_path: str | Path,
) -> dict[str, Any]:
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    frames = [load_rgb(path) for path in bundle["marked_pages"]]
    keys = [DataKey.from_mapping(value) for value in bundle["keys"]]
    verification = verify_sequence(
        frames, keys, public_key=load_public_key(public_key_path)
    )
    return {
        "valid": verification.valid,
        "signature_valid": list(verification.signature_valid),
        "chain_valid": verification.chain_valid,
        "index_valid": verification.index_valid,
        "asset_id_valid": verification.asset_id_valid,
        "failures": list(verification.failures),
    }


__all__ = ["embed_document_pages", "verify_document_pages"]
