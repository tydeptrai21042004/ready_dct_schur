from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dct_schur.config import SchurConfig
from dct_schur.engine import embed_plane, extract_evidence_candidates
from dct_schur.key import SchurKey
from .codec import (
    CodecProfile,
    DecodedPayload,
    EncodedPayload,
    PayloadType,
    codec_summary,
    decode_payload_evidence,
    encode_payload,
)


@dataclass
class DataKey:
    schur_key: SchurKey
    codec_seed: int
    payload_type: int
    codec_profile: str
    codec_version: int = 2

    @property
    def method_id(self) -> str:
        return "dct_schur_data_transport"

    @property
    def host_shape(self) -> tuple[int, int, int]:
        return self.schur_key.host_shape

    def to_dict(self) -> dict[str, Any]:
        return {
            "schur_key": self.schur_key.to_dict(),
            "codec_seed": int(self.codec_seed),
            "payload_type": int(self.payload_type),
            "codec_profile": str(self.codec_profile),
            "codec_version": int(self.codec_version),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DataKey":
        raw = dict(value)
        raw["schur_key"] = SchurKey.from_mapping(raw["schur_key"])
        return cls(**raw)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "DataKey":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def real_world_config(value: SchurConfig | Mapping[str, Any] | None = None) -> SchurConfig:
    cfg = SchurConfig.from_mapping(value)
    if cfg.resolution_adaptive_replicas_enabled:
        return cfg
    data = cfg.to_dict()
    data["resolution_adaptive_replicas_enabled"] = True
    return SchurConfig.from_mapping(data)


def embed_bytes(
    host_rgb: np.ndarray,
    payload: bytes,
    *,
    payload_type: PayloadType | int = PayloadType.RAW_BYTES,
    config: SchurConfig | Mapping[str, Any] | None = None,
    codec_seed: int | None = None,
    compress: bool = True,
    codec_profile: CodecProfile | str = CodecProfile.AUTO,
    return_metadata: bool = False,
):
    cfg = real_world_config(config)
    seed = cfg.seed if codec_seed is None else int(codec_seed)
    encoded: EncodedPayload = encode_payload(
        payload,
        payload_type=payload_type,
        codec_seed=seed,
        compress=compress,
        profile=codec_profile,
    )
    watermarked, schur_key, schur_metadata = embed_plane(
        host_rgb, encoded.bit_plane, config=cfg, return_metadata=True
    )
    key = DataKey(
        schur_key=schur_key,
        codec_seed=seed,
        payload_type=int(encoded.payload_type),
        codec_profile=encoded.profile.value,
    )
    metadata = {
        "application": "arbitrary_payload_transport",
        "payload_type": encoded.payload_type.name.lower(),
        "payload_original_bytes": encoded.original_size,
        "payload_stored_bytes": encoded.stored_size,
        "payload_compressed": encoded.compressed,
        "codec_profile": encoded.profile.value,
        "transport_codec": codec_summary(encoded.profile),
        "semantic_acceptance": "exact decode plus CRC",
        "schur": schur_metadata,
    }
    return (watermarked, key, metadata) if return_metadata else (watermarked, key)


def extract_bytes(
    possibly_attacked_rgb: np.ndarray,
    key: DataKey,
    *,
    require_crc: bool = True,
    return_metadata: bool = False,
):
    candidates, schur_metadata = extract_evidence_candidates(
        possibly_attacked_rgb, key.schur_key, return_metadata=True
    )
    attempts: list[dict[str, Any]] = []
    decoded: DecodedPayload | None = None
    selected: dict[str, Any] | None = None
    last_error: Exception | None = None
    decode_candidates = candidates if require_crc else candidates[:1]
    for candidate in decode_candidates:
        try:
            trial = decode_payload_evidence(
                candidate["evidence"],
                codec_seed=key.codec_seed,
                profile=key.codec_profile,
                require_crc=require_crc,
            )
            if int(trial.payload_type) != int(key.payload_type):
                raise ValueError(
                    f"decoded payload type {int(trial.payload_type)} does not match "
                    f"key type {key.payload_type}"
                )
            decoded = trial
            selected = candidate
            attempts.append(
                {
                    "rank": int(candidate["rank"]),
                    "candidate": str(candidate["candidate"]),
                    "crc_valid": bool(trial.crc_valid),
                    "accepted": True,
                }
            )
            break
        except Exception as exc:
            last_error = exc
            attempts.append(
                {
                    "rank": int(candidate["rank"]),
                    "candidate": str(candidate["candidate"]),
                    "crc_valid": False,
                    "accepted": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if decoded is None or selected is None:
        if last_error is not None:
            raise last_error
        raise ValueError("no Schur evidence candidate could be decoded")

    metadata = {
        "application": "arbitrary_payload_transport",
        "payload_type": decoded.payload_type.name.lower(),
        "payload_bytes": len(decoded.payload),
        "payload_compressed": decoded.compressed,
        "codec_profile": decoded.profile.value,
        "crc_valid": decoded.crc_valid,
        "viterbi_path_metric": decoded.path_metric,
        "viterbi_terminal_metric_gap": decoded.terminal_metric_gap,
        "selected_candidate": selected["candidate"],
        "selected_candidate_rank": selected["rank"],
        "candidate_attempts": attempts,
        "semantic_acceptance": "exact decode plus CRC",
        "schur": schur_metadata,
    }
    return (decoded.payload, metadata) if return_metadata else decoded.payload


def embed_text(host_rgb: np.ndarray, text: str, **kwargs: Any):
    return embed_bytes(
        host_rgb, text.encode("utf-8"), payload_type=PayloadType.UTF8_TEXT, **kwargs
    )


def extract_text(possibly_attacked_rgb: np.ndarray, key: DataKey, **kwargs: Any):
    result = extract_bytes(possibly_attacked_rgb, key, **kwargs)
    if kwargs.get("return_metadata", False):
        payload, metadata = result
        return payload.decode("utf-8"), metadata
    return result.decode("utf-8")


def embed_json(host_rgb: np.ndarray, value: Any, **kwargs: Any):
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return embed_bytes(host_rgb, canonical, payload_type=PayloadType.JSON, **kwargs)


def extract_json(possibly_attacked_rgb: np.ndarray, key: DataKey, **kwargs: Any):
    result = extract_bytes(possibly_attacked_rgb, key, **kwargs)
    if kwargs.get("return_metadata", False):
        payload, metadata = result
        return json.loads(payload.decode("utf-8")), metadata
    return json.loads(result.decode("utf-8"))


__all__ = [
    "DataKey", "real_world_config", "embed_bytes", "extract_bytes",
    "embed_text", "extract_text", "embed_json", "extract_json",
]
