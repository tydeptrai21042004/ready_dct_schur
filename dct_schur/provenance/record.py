from __future__ import annotations

"""Compact signed soft-binding records for Schur provenance transport.

The 152-byte record carries an asset UUID and the full SHA-256 digest of an
external provenance manifest.  The manifest itself remains in a C2PA/JPEG Trust
repository or file metadata; the invisible watermark acts as a durable lookup
binding after metadata stripping.  Ed25519 authenticates the token.
"""

from dataclasses import dataclass
from enum import IntEnum, IntFlag
import hashlib
import struct
import time
import uuid
from typing import Any

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

MAGIC = b"DSPV"
VERSION = 2
# 88 signed bytes + 64-byte Ed25519 signature = 152 bytes.
UNSIGNED_STRUCT = struct.Struct(">4sBBH16sI8s32sHH16s")
SIGNATURE_SIZE = 64
RECORD_SIZE = UNSIGNED_STRUCT.size + SIGNATURE_SIZE


class MediaType(IntEnum):
    IMAGE = 1
    VIDEO_FRAME = 2
    DOCUMENT_PAGE = 3
    SCREEN_CAPTURE = 4
    OTHER_VISUAL = 255


class ProvenanceFlags(IntFlag):
    NONE = 0
    CAMERA_CAPTURED = 1 << 0
    AI_GENERATED = 1 << 1
    EDITED = 1 << 2
    RIGHTS_RESERVED = 1 << 3
    PUBLIC_INTEREST = 1 << 4
    SENSITIVE = 1 << 5


def _sha256(value: bytes | bytearray | memoryview | np.ndarray | None) -> bytes:
    if value is None:
        return bytes(32)
    if isinstance(value, np.ndarray):
        data = np.ascontiguousarray(value).tobytes()
    else:
        data = bytes(value)
    return hashlib.sha256(data).digest()


def public_key_bytes(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def issuer_key_id(public_key: Ed25519PublicKey) -> bytes:
    return hashlib.sha256(public_key_bytes(public_key)).digest()[:8]


@dataclass(frozen=True)
class ProvenanceRecord:
    asset_id: uuid.UUID
    issued_at: int
    issuer_id: bytes
    manifest_digest: bytes
    media_type: MediaType = MediaType.IMAGE
    flags: ProvenanceFlags = ProvenanceFlags.NONE
    sequence_index: int = 0
    sequence_count: int = 1
    previous_record_digest: bytes = bytes(16)
    signature: bytes = bytes(SIGNATURE_SIZE)

    def _unsigned_bytes(self) -> bytes:
        issuer = bytes(self.issuer_id)
        manifest = bytes(self.manifest_digest)
        previous = bytes(self.previous_record_digest)
        if len(issuer) != 8:
            raise ValueError("issuer_id must contain exactly 8 bytes")
        if len(manifest) != 32:
            raise ValueError("manifest_digest must contain a full SHA-256 value")
        if len(previous) != 16:
            raise ValueError("previous_record_digest must contain exactly 16 bytes")
        if not (0 <= int(self.issued_at) <= 0xFFFFFFFF):
            raise ValueError("issued_at is outside uint32 range")
        if not (0 <= int(self.sequence_index) <= 0xFFFF):
            raise ValueError("sequence_index is outside uint16 range")
        if not (1 <= int(self.sequence_count) <= 0xFFFF):
            raise ValueError("sequence_count is outside uint16 range")
        return UNSIGNED_STRUCT.pack(
            MAGIC,
            VERSION,
            int(self.media_type),
            int(self.flags),
            self.asset_id.bytes,
            int(self.issued_at),
            issuer,
            manifest,
            int(self.sequence_index),
            int(self.sequence_count),
            previous,
        )

    def to_bytes(self) -> bytes:
        signature = bytes(self.signature)
        if len(signature) != SIGNATURE_SIZE:
            raise ValueError("signature must contain exactly 64 bytes")
        return self._unsigned_bytes() + signature

    def signed(self, private_key: Ed25519PrivateKey) -> "ProvenanceRecord":
        signature = private_key.sign(self._unsigned_bytes())
        return ProvenanceRecord(
            asset_id=self.asset_id,
            issued_at=self.issued_at,
            issuer_id=self.issuer_id,
            manifest_digest=self.manifest_digest,
            media_type=self.media_type,
            flags=self.flags,
            sequence_index=self.sequence_index,
            sequence_count=self.sequence_count,
            previous_record_digest=self.previous_record_digest,
            signature=signature,
        )

    def verify(self, public_key: Ed25519PublicKey) -> bool:
        if self.issuer_id != issuer_key_id(public_key):
            return False
        try:
            public_key.verify(bytes(self.signature), self._unsigned_bytes())
            return True
        except InvalidSignature:
            return False

    @property
    def record_digest(self) -> bytes:
        return hashlib.sha256(self.to_bytes()).digest()

    def summary(self) -> dict[str, Any]:
        return {
            "asset_id": str(self.asset_id),
            "issued_at": int(self.issued_at),
            "issuer_id_hex": self.issuer_id.hex(),
            "manifest_sha256": self.manifest_digest.hex(),
            "media_type": self.media_type.name.lower(),
            "flags": int(self.flags),
            "sequence_index": int(self.sequence_index),
            "sequence_count": int(self.sequence_count),
            "previous_record_digest_hex": self.previous_record_digest.hex(),
            "record_digest_hex": self.record_digest.hex(),
            "binding_role": "external_manifest_soft_binding",
        }

    @classmethod
    def from_bytes(cls, value: bytes) -> "ProvenanceRecord":
        raw = bytes(value)
        if len(raw) != RECORD_SIZE:
            raise ValueError(f"provenance record must be {RECORD_SIZE} bytes")
        (
            magic,
            version,
            media_type,
            flags,
            asset_bytes,
            issued_at,
            issuer,
            manifest,
            sequence_index,
            sequence_count,
            previous,
        ) = UNSIGNED_STRUCT.unpack(raw[: UNSIGNED_STRUCT.size])
        if magic != MAGIC:
            raise ValueError("invalid provenance record magic")
        if version != VERSION:
            raise ValueError(f"unsupported provenance record version {version}")
        return cls(
            asset_id=uuid.UUID(bytes=asset_bytes),
            issued_at=int(issued_at),
            issuer_id=bytes(issuer),
            manifest_digest=bytes(manifest),
            media_type=MediaType(int(media_type)),
            flags=ProvenanceFlags(int(flags)),
            sequence_index=int(sequence_index),
            sequence_count=int(sequence_count),
            previous_record_digest=bytes(previous),
            signature=raw[UNSIGNED_STRUCT.size :],
        )


def create_provenance_record(
    *,
    private_key: Ed25519PrivateKey,
    source_content: bytes | np.ndarray | None = None,
    manifest_bytes: bytes | None = None,
    manifest_digest: bytes | None = None,
    asset_id: uuid.UUID | None = None,
    issued_at: int | None = None,
    media_type: MediaType = MediaType.IMAGE,
    flags: ProvenanceFlags = ProvenanceFlags.NONE,
    sequence_index: int = 0,
    sequence_count: int = 1,
    previous_record: ProvenanceRecord | None = None,
) -> ProvenanceRecord:
    if manifest_bytes is not None and manifest_digest is not None:
        raise ValueError("provide manifest_bytes or manifest_digest, not both")
    if manifest_digest is not None:
        manifest_hash = bytes(manifest_digest)
        if len(manifest_hash) != 32:
            raise ValueError("manifest_digest must contain exactly 32 bytes")
    elif manifest_bytes is not None:
        manifest_hash = _sha256(manifest_bytes)
    else:
        # Standalone mode: bind to a source digest when no external manifest is
        # available.  Production provenance should supply the manifest bytes.
        manifest_hash = _sha256(source_content)

    public_key = private_key.public_key()
    previous = previous_record.record_digest[:16] if previous_record else bytes(16)
    record = ProvenanceRecord(
        asset_id=asset_id or uuid.uuid4(),
        issued_at=int(time.time()) if issued_at is None else int(issued_at),
        issuer_id=issuer_key_id(public_key),
        manifest_digest=manifest_hash,
        media_type=media_type,
        flags=flags,
        sequence_index=int(sequence_index),
        sequence_count=int(sequence_count),
        previous_record_digest=previous,
    )
    return record.signed(private_key)


def generate_signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


__all__ = [
    "MediaType",
    "ProvenanceFlags",
    "ProvenanceRecord",
    "RECORD_SIZE",
    "create_provenance_record",
    "generate_signing_key",
    "issuer_key_id",
    "public_key_bytes",
]
