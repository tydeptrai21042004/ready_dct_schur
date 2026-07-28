from __future__ import annotations

from dct_schur.io import load_rgb
from dct_schur.provenance import create_and_embed, embed_sequence, extract_record, verify_sequence
from dct_schur.provenance.record import generate_signing_key


def test_signed_provenance_roundtrip() -> None:
    host = load_rgb("data/host/lenna.bmp")
    private_key = generate_signing_key()
    marked, key, original = create_and_embed(
        host, private_key=private_key, manifest_bytes=b"manifest-v1"
    )
    recovered, metadata = extract_record(
        marked, key, public_key=private_key.public_key(), return_metadata=True
    )
    assert recovered.asset_id == original.asset_id
    assert metadata["signature_valid"] is True


def test_sequence_chain_roundtrip() -> None:
    frame = load_rgb("data/host/lenna.bmp")
    frames = [frame, frame.copy()]
    private_key = generate_signing_key()
    marked, keys, _records = embed_sequence(frames, private_key=private_key)
    verification = verify_sequence(marked, keys, public_key=private_key.public_key())
    assert verification.valid
    assert verification.chain_valid
