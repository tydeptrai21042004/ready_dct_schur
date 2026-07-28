from __future__ import annotations

from dataclasses import replace
import numpy as np

from dct_schur import SchurConfig, embed_plane, extract_plane, philosophy_summary
from dct_schur.engine.evidence import _witness_trust
from dct_schur.io import load_binary_plane, load_rgb
from evaluation.metrics import nc, psnr


def test_clean_roundtrip_above_50_db() -> None:
    host = load_rgb("data/host/lenna.bmp")
    payload = load_binary_plane("data/watermark/wm.png")
    marked, key, metadata = embed_plane(host, payload, return_metadata=True)
    recovered = extract_plane(marked, key)
    assert psnr(host, marked) > 50.0
    assert nc(payload, recovered) == 1.0
    assert metadata["spectrum_preserved_float"]
    assert metadata["trace_preserved_float"]
    assert metadata["determinant_preserved_float"]
    assert metadata["coset_projection_energy_ratio"] <= 1.0


def test_identity_witness_ignores_global_gain_and_marks_local_damage() -> None:
    cfg = SchurConfig(witness_log_sigma=0.5, witness_floor=0.3)
    reference = np.ones(16)
    current = np.ones(16) * 1.2
    current[3] = 4.0
    trust, ratio = _witness_trust(current, reference, cfg)
    assert np.median(trust) > 0.99
    assert trust[3] < 0.5
    assert ratio[3] == 4.0


def test_core_philosophy_is_operational() -> None:
    summary = philosophy_summary()
    assert "spectral identity" in summary["thesis"]
    assert len(summary["principles"]) == 4
    rules = " ".join(item["implementation_rule"] for item in summary["principles"])
    assert "strict-upper" in rules
    assert "CRC" in rules
