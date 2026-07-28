from __future__ import annotations

"""Error-corrected 4096-bit transport codec for SP-SCQIM.

Two terminated convolutional-code profiles are provided:

* ``robust_r13``: rate 1/3, up to 155 stored bytes.  This is the default for
  signed provenance tokens and prioritizes exact recovery.
* ``capacity_r12``: rate 1/2, up to 241 stored bytes for larger text/JSON data.

Both profiles use seeded bit interleaving, a typed header, optional DEFLATE
compression, and CRC-32 validation.  They decode directly from Schur soft
evidence and do not assume a spatially smooth logo.
"""

from dataclasses import dataclass
from enum import IntEnum, Enum
import struct
import zlib
from typing import Any

import numpy as np

RAW_BITS = 4096
CONSTRAINT_LENGTH = 7
TAIL_BITS = CONSTRAINT_LENGTH - 1
MAGIC = b"DSCP"
VERSION = 2
FLAG_COMPRESSED = 0x01
HEADER_PREFIX = struct.Struct(">4sBBBBH")
HEADER = struct.Struct(">4sBBBBHI")
HEADER_SIZE = HEADER.size
_STATE_COUNT = 1 << (CONSTRAINT_LENGTH - 1)
_REGISTER_MASK = (1 << CONSTRAINT_LENGTH) - 1
_STATE_MASK = _STATE_COUNT - 1


class PayloadType(IntEnum):
    RAW_BYTES = 0
    UTF8_TEXT = 1
    JSON = 2
    PROVENANCE = 3
    URI = 4
    FILE_DIGEST = 5


class CodecProfile(str, Enum):
    AUTO = "auto"
    ROBUST_R13 = "robust_r13"
    CAPACITY_R12 = "capacity_r12"


@dataclass(frozen=True)
class _ProfileSpec:
    profile: CodecProfile
    generators: tuple[int, ...]
    input_bits: int
    code_bits: int
    pad_bits: int
    data_bits: int
    packed_data_bytes: int
    spare_data_bits: int
    max_payload_bytes: int


def _profile_spec(profile: CodecProfile | str) -> _ProfileSpec:
    normalized = CodecProfile(profile)
    if normalized == CodecProfile.AUTO:
        raise ValueError("AUTO must be resolved before requesting a profile spec")
    generators = (0o171, 0o133, 0o165) if normalized == CodecProfile.ROBUST_R13 else (0o171, 0o133)
    outputs = len(generators)
    input_bits = RAW_BITS // outputs
    code_bits = input_bits * outputs
    pad_bits = RAW_BITS - code_bits
    data_bits = input_bits - TAIL_BITS
    packed_data_bytes = data_bits // 8
    spare_data_bits = data_bits - packed_data_bytes * 8
    return _ProfileSpec(
        profile=normalized,
        generators=generators,
        input_bits=input_bits,
        code_bits=code_bits,
        pad_bits=pad_bits,
        data_bits=data_bits,
        packed_data_bytes=packed_data_bytes,
        spare_data_bits=spare_data_bits,
        max_payload_bytes=packed_data_bytes - HEADER_SIZE,
    )


ROBUST_SPEC = _profile_spec(CodecProfile.ROBUST_R13)
CAPACITY_SPEC = _profile_spec(CodecProfile.CAPACITY_R12)
MAX_PAYLOAD_BYTES = CAPACITY_SPEC.max_payload_bytes
ROBUST_MAX_PAYLOAD_BYTES = ROBUST_SPEC.max_payload_bytes


@dataclass(frozen=True)
class EncodedPayload:
    bit_plane: np.ndarray
    payload_type: PayloadType
    profile: CodecProfile
    original_size: int
    stored_size: int
    compressed: bool
    codec_seed: int


@dataclass(frozen=True)
class DecodedPayload:
    payload: bytes
    payload_type: PayloadType
    profile: CodecProfile
    compressed: bool
    crc_valid: bool
    codec_seed: int
    path_metric: float
    terminal_metric_gap: float

    def text(self) -> str:
        return self.payload.decode("utf-8")


def _parity(value: int) -> int:
    return int(value.bit_count() & 1)


def _trellis(generators: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    next_state = np.empty((_STATE_COUNT, 2), dtype=np.int16)
    outputs = np.empty((_STATE_COUNT, 2, len(generators)), dtype=np.uint8)
    for state in range(_STATE_COUNT):
        for bit in (0, 1):
            register = ((state << 1) | bit) & _REGISTER_MASK
            next_state[state, bit] = register & _STATE_MASK
            outputs[state, bit] = [_parity(register & g) for g in generators]
    signs = np.where(outputs > 0, 1.0, -1.0)
    return next_state, outputs, signs


_TRELLIS = {
    CodecProfile.ROBUST_R13: _trellis(ROBUST_SPEC.generators),
    CodecProfile.CAPACITY_R12: _trellis(CAPACITY_SPEC.generators),
}


def _predecessor_table(next_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the two predecessor transitions for every trellis state."""
    states = np.empty((_STATE_COUNT, 2), dtype=np.int16)
    bits = np.empty((_STATE_COUNT, 2), dtype=np.uint8)
    counts = np.zeros(_STATE_COUNT, dtype=np.uint8)
    for state in range(_STATE_COUNT):
        for bit in (0, 1):
            nxt = int(next_state[state, bit])
            slot = int(counts[nxt])
            states[nxt, slot] = state
            bits[nxt, slot] = bit
            counts[nxt] += 1
    if not np.all(counts == 2):
        raise AssertionError("convolutional trellis must have two predecessors per state")
    return states, bits


_PREDECESSORS = {
    profile: _predecessor_table(trellis[0])
    for profile, trellis in _TRELLIS.items()
}


def _convolutional_encode(input_bits: np.ndarray, spec: _ProfileSpec) -> np.ndarray:
    bits = np.asarray(input_bits, dtype=np.uint8).reshape(-1) & 1
    if bits.size != spec.input_bits:
        raise ValueError(f"expected {spec.input_bits} input bits; got {bits.size}")
    next_state, output_bits, _ = _TRELLIS[spec.profile]
    state = 0
    coded = np.empty(spec.code_bits, dtype=np.uint8)
    cursor = 0
    outputs = len(spec.generators)
    for bit in bits:
        coded[cursor : cursor + outputs] = output_bits[state, int(bit)]
        state = int(next_state[state, int(bit)])
        cursor += outputs
    if state != 0:
        raise AssertionError("terminated convolutional encoder did not end in state zero")
    if spec.pad_bits:
        coded = np.concatenate([coded, np.zeros(spec.pad_bits, dtype=np.uint8)])
    return coded


def _viterbi_decode(soft_code: np.ndarray, spec: _ProfileSpec) -> tuple[np.ndarray, float, float]:
    evidence = np.asarray(soft_code, dtype=np.float64).reshape(-1)
    if evidence.size != RAW_BITS:
        raise ValueError(f"expected {RAW_BITS} soft code values; got {evidence.size}")
    outputs = len(spec.generators)
    observations = evidence[: spec.code_bits].reshape(spec.input_bits, outputs)
    _, _, output_signs = _TRELLIS[spec.profile]
    pred_states, pred_bits = _PREDECESSORS[spec.profile]
    pred_signs = output_signs[pred_states, pred_bits]

    metric = np.full(_STATE_COUNT, -np.inf, dtype=np.float64)
    metric[0] = 0.0
    predecessor = np.zeros((spec.input_bits, _STATE_COUNT), dtype=np.int16)
    decision = np.zeros((spec.input_bits, _STATE_COUNT), dtype=np.uint8)

    for time_index, observed in enumerate(observations):
        branch_metric = np.einsum("nko,o->nk", pred_signs, observed)
        candidate_metric = metric[pred_states] + branch_metric
        choice = np.argmax(candidate_metric, axis=1)
        state_index = np.arange(_STATE_COUNT)
        metric = candidate_metric[state_index, choice]
        predecessor[time_index] = pred_states[state_index, choice]
        decision[time_index] = pred_bits[state_index, choice]

    finite = np.sort(metric[np.isfinite(metric)])
    terminal_gap = float(finite[-1] - finite[-2]) if finite.size >= 2 else float("inf")
    terminal_state = 0 if np.isfinite(metric[0]) else int(np.argmax(metric))
    final_metric = float(metric[terminal_state])

    decoded = np.empty(spec.input_bits, dtype=np.uint8)
    state = terminal_state
    for time_index in range(spec.input_bits - 1, -1, -1):
        decoded[time_index] = decision[time_index, state]
        state = int(predecessor[time_index, state])
    return decoded, final_metric, terminal_gap


def _interleaver(seed: int, profile: CodecProfile) -> np.ndarray:
    salt = 0xD5C0A11E if profile == CodecProfile.CAPACITY_R12 else 0x13C0DEC0
    rng = np.random.default_rng(int(seed) ^ salt)
    return rng.permutation(RAW_BITS).astype(np.int32)


def _prepare_payload(payload: bytes, compress: bool) -> tuple[bytes, bool]:
    raw = bytes(payload)
    if not compress or not raw:
        return raw, False
    compressed = zlib.compress(raw, level=9)
    if len(compressed) + 2 < len(raw):
        return compressed, True
    return raw, False


def _resolve_profile(profile: CodecProfile | str, stored_size: int) -> _ProfileSpec:
    normalized = CodecProfile(profile)
    if normalized == CodecProfile.AUTO:
        if stored_size <= ROBUST_SPEC.max_payload_bytes:
            return ROBUST_SPEC
        return CAPACITY_SPEC
    spec = _profile_spec(normalized)
    if stored_size > spec.max_payload_bytes:
        raise ValueError(
            f"{normalized.value} supports at most {spec.max_payload_bytes} stored bytes; "
            f"received {stored_size}"
        )
    return spec


def encode_payload(
    payload: bytes,
    *,
    payload_type: PayloadType | int = PayloadType.RAW_BYTES,
    codec_seed: int = 2026,
    compress: bool = True,
    profile: CodecProfile | str = CodecProfile.AUTO,
) -> EncodedPayload:
    kind = PayloadType(int(payload_type))
    original = bytes(payload)
    stored, compressed = _prepare_payload(original, compress)
    spec = _resolve_profile(profile, len(stored))
    if len(stored) > spec.max_payload_bytes:
        raise ValueError(
            f"payload requires {len(stored)} bytes after compression; "
            f"maximum for {spec.profile.value} is {spec.max_payload_bytes} bytes"
        )

    flags = FLAG_COMPRESSED if compressed else 0
    prefix = HEADER_PREFIX.pack(MAGIC, VERSION, int(kind), flags, 0, len(stored))
    checksum = zlib.crc32(prefix + stored) & 0xFFFFFFFF
    header = HEADER.pack(MAGIC, VERSION, int(kind), flags, 0, len(stored), checksum)
    packed = header + stored
    packed += bytes(spec.packed_data_bytes - len(packed))

    data_bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big")
    input_bits = np.concatenate(
        [
            data_bits,
            np.zeros(spec.spare_data_bits, dtype=np.uint8),
            np.zeros(TAIL_BITS, dtype=np.uint8),
        ]
    )
    code = _convolutional_encode(input_bits, spec)
    permutation = _interleaver(codec_seed, spec.profile)
    transmitted = code[permutation]
    plane = (transmitted.reshape(64, 64) * 255).astype(np.uint8)
    return EncodedPayload(
        bit_plane=plane,
        payload_type=kind,
        profile=spec.profile,
        original_size=len(original),
        stored_size=len(stored),
        compressed=compressed,
        codec_seed=int(codec_seed),
    )


def decode_payload_evidence(
    evidence_plane: np.ndarray,
    *,
    codec_seed: int = 2026,
    profile: CodecProfile | str = CodecProfile.CAPACITY_R12,
    require_crc: bool = True,
) -> DecodedPayload:
    spec = _profile_spec(CodecProfile(profile))
    evidence = np.asarray(evidence_plane, dtype=np.float64).reshape(-1)
    if evidence.size != RAW_BITS:
        raise ValueError(f"evidence plane must contain {RAW_BITS} values")
    permutation = _interleaver(codec_seed, spec.profile)
    code_evidence = np.empty(RAW_BITS, dtype=np.float64)
    code_evidence[permutation] = evidence
    decoded_bits, path_metric, terminal_gap = _viterbi_decode(code_evidence, spec)

    packed_bits = decoded_bits[: spec.packed_data_bytes * 8]
    packed = np.packbits(packed_bits, bitorder="big").tobytes()
    magic, version, kind, flags, reserved, length, checksum = HEADER.unpack(
        packed[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ValueError("decoded payload magic is invalid")
    if version != VERSION:
        raise ValueError(f"unsupported payload codec version {version}")
    if reserved != 0:
        raise ValueError("decoded payload header has nonzero reserved bits")
    if length > spec.max_payload_bytes:
        raise ValueError("decoded payload length exceeds transport capacity")
    stored = packed[HEADER_SIZE : HEADER_SIZE + length]
    prefix = HEADER_PREFIX.pack(magic, version, kind, flags, reserved, length)
    actual_checksum = zlib.crc32(prefix + stored) & 0xFFFFFFFF
    crc_valid = actual_checksum == checksum
    if require_crc and not crc_valid:
        raise ValueError("decoded payload failed CRC-32 validation")

    compressed = bool(flags & FLAG_COMPRESSED)
    payload = zlib.decompress(stored) if compressed else stored
    return DecodedPayload(
        payload=payload,
        payload_type=PayloadType(int(kind)),
        profile=spec.profile,
        compressed=compressed,
        crc_valid=crc_valid,
        codec_seed=int(codec_seed),
        path_metric=path_metric,
        terminal_metric_gap=terminal_gap,
    )


def codec_summary(profile: CodecProfile | str | None = None) -> dict[str, Any]:
    profiles = [ROBUST_SPEC, CAPACITY_SPEC]
    if profile is not None and CodecProfile(profile) != CodecProfile.AUTO:
        profiles = [_profile_spec(CodecProfile(profile))]
    return {
        "raw_bits": RAW_BITS,
        "interleaved": True,
        "crc": "CRC-32",
        "profiles": {
            spec.profile.value: {
                "rate": f"1/{len(spec.generators)}",
                "constraint_length": CONSTRAINT_LENGTH,
                "generators_octal": [oct(g) for g in spec.generators],
                "max_payload_bytes": spec.max_payload_bytes,
            }
            for spec in profiles
        },
    }


__all__ = [
    "PayloadType",
    "CodecProfile",
    "EncodedPayload",
    "DecodedPayload",
    "MAX_PAYLOAD_BYTES",
    "ROBUST_MAX_PAYLOAD_BYTES",
    "encode_payload",
    "decode_payload_evidence",
    "codec_summary",
]
