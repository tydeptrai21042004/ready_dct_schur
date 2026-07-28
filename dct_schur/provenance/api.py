from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import uuid

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from dct_schur.config import SchurConfig
from dct_schur.transport.api import DataKey, embed_bytes, extract_bytes
from dct_schur.transport.codec import CodecProfile, PayloadType
from .record import MediaType, ProvenanceFlags, ProvenanceRecord, create_provenance_record


@dataclass(frozen=True)
class SequenceVerification:
    valid: bool
    records: tuple[ProvenanceRecord, ...]
    signature_valid: tuple[bool, ...]
    chain_valid: bool
    index_valid: bool
    asset_id_valid: bool
    failures: tuple[str, ...]


def embed_record(
    host_rgb: np.ndarray,
    record: ProvenanceRecord,
    **kwargs: Any,
):
    return embed_bytes(
        host_rgb,
        record.to_bytes(),
        payload_type=PayloadType.PROVENANCE,
        compress=False,
        codec_profile=CodecProfile.ROBUST_R13,
        **kwargs,
    )


def extract_record(
    possibly_attacked_rgb: np.ndarray,
    key: DataKey,
    *,
    public_key: Ed25519PublicKey | None = None,
    return_metadata: bool = False,
):
    payload, metadata = extract_bytes(
        possibly_attacked_rgb, key, return_metadata=True
    )
    record = ProvenanceRecord.from_bytes(payload)
    output = dict(metadata)
    output["provenance"] = record.summary()
    if public_key is not None:
        output["signature_valid"] = record.verify(public_key)
    return (record, output) if return_metadata else record


def create_and_embed(
    host_rgb: np.ndarray,
    *,
    private_key: Ed25519PrivateKey,
    manifest_bytes: bytes | None = None,
    asset_id: uuid.UUID | None = None,
    flags: ProvenanceFlags = ProvenanceFlags.NONE,
    media_type: MediaType = MediaType.IMAGE,
    config: SchurConfig | Mapping[str, Any] | None = None,
    return_metadata: bool = False,
):
    record = create_provenance_record(
        private_key=private_key,
        source_content=np.asarray(host_rgb, dtype=np.uint8),
        manifest_bytes=manifest_bytes,
        asset_id=asset_id,
        flags=flags,
        media_type=media_type,
    )
    watermarked, key, metadata = embed_record(
        host_rgb, record, config=config, return_metadata=True
    )
    output = dict(metadata)
    output["provenance"] = record.summary()
    return (watermarked, key, record, output) if return_metadata else (
        watermarked, key, record
    )


def embed_sequence(
    frames: Iterable[np.ndarray],
    *,
    private_key: Ed25519PrivateKey,
    manifest_bytes: bytes | None = None,
    asset_id: uuid.UUID | None = None,
    flags: ProvenanceFlags = ProvenanceFlags.NONE,
    media_type: MediaType = MediaType.VIDEO_FRAME,
    config: SchurConfig | Mapping[str, Any] | None = None,
) -> tuple[list[np.ndarray], list[DataKey], list[ProvenanceRecord]]:
    frame_list = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    if not frame_list:
        raise ValueError("at least one frame is required")
    sequence_asset_id = asset_id or uuid.uuid4()
    watermarked: list[np.ndarray] = []
    keys: list[DataKey] = []
    records: list[ProvenanceRecord] = []
    previous: ProvenanceRecord | None = None
    for index, frame in enumerate(frame_list):
        record = create_provenance_record(
            private_key=private_key,
            source_content=frame,
            manifest_bytes=manifest_bytes,
            asset_id=sequence_asset_id,
            flags=flags,
            media_type=media_type,
            sequence_index=index,
            sequence_count=len(frame_list),
            previous_record=previous,
        )
        marked, key = embed_record(frame, record, config=config)
        watermarked.append(marked)
        keys.append(key)
        records.append(record)
        previous = record
    return watermarked, keys, records


def verify_sequence(
    frames: Iterable[np.ndarray],
    keys: Iterable[DataKey],
    *,
    public_key: Ed25519PublicKey,
) -> SequenceVerification:
    frame_list = list(frames)
    key_list = list(keys)
    if len(frame_list) != len(key_list):
        raise ValueError("frames and keys must have the same length")
    records: list[ProvenanceRecord] = []
    signatures: list[bool] = []
    failures: list[str] = []
    for index, (frame, key) in enumerate(zip(frame_list, key_list)):
        try:
            record = extract_record(frame, key)
            records.append(record)
            valid_signature = record.verify(public_key)
            signatures.append(valid_signature)
            if not valid_signature:
                failures.append(f"frame {index}: invalid signature")
        except Exception as exc:
            failures.append(f"frame {index}: extraction failed: {exc}")
    if len(records) != len(frame_list):
        return SequenceVerification(
            False, tuple(records), tuple(signatures), False, False, False, tuple(failures)
        )
    first_asset = records[0].asset_id
    asset_valid = all(record.asset_id == first_asset for record in records)
    index_valid = all(
        record.sequence_index == index and record.sequence_count == len(records)
        for index, record in enumerate(records)
    )
    chain_valid = records[0].previous_record_digest == bytes(16)
    for previous, current in zip(records, records[1:]):
        chain_valid = chain_valid and (
            current.previous_record_digest == previous.record_digest[:16]
        )
    if not asset_valid:
        failures.append("asset identifiers are inconsistent")
    if not index_valid:
        failures.append("sequence indexes or counts are inconsistent")
    if not chain_valid:
        failures.append("record hash chain is broken")
    valid = all(signatures) and asset_valid and index_valid and chain_valid
    return SequenceVerification(
        valid, tuple(records), tuple(signatures), chain_valid,
        index_valid, asset_valid, tuple(failures)
    )


__all__ = [
    "SequenceVerification", "embed_record", "extract_record",
    "create_and_embed", "embed_sequence", "verify_sequence",
]
