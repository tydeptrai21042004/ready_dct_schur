from __future__ import annotations

from pathlib import Path

from dct_schur.io import load_rgb
from dct_schur.transport import DataKey, embed_json, embed_text, extract_json, extract_text


def test_text_and_json_roundtrip(tmp_path: Path) -> None:
    host = load_rgb("data/host/lenna.bmp")
    marked, key = embed_text(host, "Invariant and relation")
    assert extract_text(marked, key) == "Invariant and relation"

    key_path = tmp_path / "data_key.json"
    key.save(key_path)
    loaded = DataKey.load(key_path)
    assert extract_text(marked, loaded) == "Invariant and relation"

    value = {"asset": "demo", "version": 3, "valid": True}
    marked_json, json_key = embed_json(host, value)
    assert extract_json(marked_json, json_key) == value
