from __future__ import annotations

from dct_schur import SchurConfig, embed_plane, extract_plane
from dct_schur.io import load_binary_plane, load_rgb
from evaluation.attacks import apply_attack, get_attack_suite
from evaluation.metrics import nc


def test_witness_profile_handles_occlusion() -> None:
    host = load_rgb("data/host/lenna.bmp")
    payload = load_binary_plane("data/watermark/wm.png")
    config = SchurConfig(candidate_search_enabled=False)
    marked, key = embed_plane(host, payload, config=config)
    attack = next(
        item for item in get_attack_suite("common")
        if item.category == "occlusion" and "fraction" in item.attack_id
    )
    recovered = extract_plane(apply_attack(marked, attack), key)
    assert nc(payload, recovered) > 0.85
