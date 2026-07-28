from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from dct_schur.config import SchurConfig
from dct_schur.io import load_rgb, save_rgb
from dct_schur.provenance import MediaType, ProvenanceFlags, create_and_embed, extract_record
from dct_schur.provenance.keys import load_private_key, load_public_key
from dct_schur.transport import DataKey
from evaluation.metrics import psnr


@dataclass(frozen=True)
class ProvenancePipelineResult:
    output_image: Path
    key_file: Path
    record_file: Path
    psnr_db: float
    asset_id: str


def embed_provenance(
    host_path: str | Path,
    private_key_path: str | Path,
    output_image_path: str | Path,
    watermark_key_path: str | Path,
    record_json_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    flags: ProvenanceFlags = ProvenanceFlags.NONE,
    media_type: MediaType = MediaType.IMAGE,
    config: SchurConfig | Mapping[str, Any] | None = None,
) -> ProvenancePipelineResult:
    host = load_rgb(host_path)
    private_key = load_private_key(private_key_path)
    manifest = Path(manifest_path).read_bytes() if manifest_path is not None else None
    marked, key, record = create_and_embed(
        host,
        private_key=private_key,
        manifest_bytes=manifest,
        flags=flags,
        media_type=media_type,
        config=config,
    )
    save_rgb(output_image_path, marked)
    key.save(watermark_key_path)
    record_target = Path(record_json_path)
    record_target.parent.mkdir(parents=True, exist_ok=True)
    record_target.write_text(json.dumps(record.summary(), indent=2), encoding="utf-8")
    return ProvenancePipelineResult(
        Path(output_image_path), Path(watermark_key_path), record_target,
        psnr(host, marked), str(record.asset_id),
    )


def verify_provenance(
    image_path: str | Path,
    watermark_key_path: str | Path,
    public_key_path: str | Path,
) -> dict[str, Any]:
    image = load_rgb(image_path)
    key = DataKey.load(watermark_key_path)
    public_key = load_public_key(public_key_path)
    record, metadata = extract_record(image, key, public_key=public_key, return_metadata=True)
    return {
        "valid": bool(metadata.get("signature_valid", False)),
        "record": record.summary(),
        "decoder": metadata,
    }


__all__ = ["ProvenancePipelineResult", "embed_provenance", "verify_provenance"]
