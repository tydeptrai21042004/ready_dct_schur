from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from dct_schur.config import SchurConfig
from dct_schur.io import load_rgb, save_rgb
from dct_schur.transport import (
    CodecProfile, DataKey, PayloadType, embed_bytes, embed_json, embed_text,
    extract_bytes, extract_json, extract_text,
)
from evaluation.metrics import psnr


@dataclass(frozen=True)
class DataPipelineResult:
    output_image: Path
    key_file: Path
    payload_bytes: int
    psnr_db: float


def embed_payload_file(
    host_path: str | Path,
    payload_path: str | Path,
    output_image_path: str | Path,
    key_path: str | Path,
    *,
    config: SchurConfig | Mapping[str, Any] | None = None,
    profile: CodecProfile | str = CodecProfile.AUTO,
) -> DataPipelineResult:
    host = load_rgb(host_path)
    payload = Path(payload_path).read_bytes()
    marked, key = embed_bytes(host, payload, config=config, codec_profile=profile)
    save_rgb(output_image_path, marked)
    key.save(key_path)
    return DataPipelineResult(Path(output_image_path), Path(key_path), len(payload), psnr(host, marked))


def extract_payload_file(
    image_path: str | Path,
    key_path: str | Path,
    output_payload_path: str | Path,
) -> Path:
    image = load_rgb(image_path)
    key = DataKey.load(key_path)
    payload = extract_bytes(image, key)
    target = Path(output_payload_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def embed_text_value(
    host_path: str | Path,
    text: str,
    output_image_path: str | Path,
    key_path: str | Path,
    *,
    config: SchurConfig | Mapping[str, Any] | None = None,
) -> DataPipelineResult:
    host = load_rgb(host_path)
    marked, key = embed_text(host, text, config=config)
    save_rgb(output_image_path, marked)
    key.save(key_path)
    return DataPipelineResult(Path(output_image_path), Path(key_path), len(text.encode("utf-8")), psnr(host, marked))


def extract_text_value(image_path: str | Path, key_path: str | Path) -> str:
    return extract_text(load_rgb(image_path), DataKey.load(key_path))


def embed_json_file(
    host_path: str | Path,
    json_path: str | Path,
    output_image_path: str | Path,
    key_path: str | Path,
    *,
    config: SchurConfig | Mapping[str, Any] | None = None,
) -> DataPipelineResult:
    value = json.loads(Path(json_path).read_text(encoding="utf-8"))
    host = load_rgb(host_path)
    marked, key = embed_json(host, value, config=config)
    save_rgb(output_image_path, marked)
    key.save(key_path)
    size = len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return DataPipelineResult(Path(output_image_path), Path(key_path), size, psnr(host, marked))


def extract_json_file(
    image_path: str | Path,
    key_path: str | Path,
    output_json_path: str | Path,
) -> Path:
    value = extract_json(load_rgb(image_path), DataKey.load(key_path))
    target = Path(output_json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


__all__ = [
    "DataPipelineResult", "embed_payload_file", "extract_payload_file",
    "embed_text_value", "extract_text_value", "embed_json_file", "extract_json_file",
]
