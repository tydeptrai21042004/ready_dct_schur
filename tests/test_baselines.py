from __future__ import annotations

from baselines import ALL_BASELINE_IDS, embed_baseline, extract_baseline, get_baseline_spec
from benchmarking.registry import DCT_SCHUR_ID, all_method_specs, resolve_methods
from benchmarking.runner import prepare_payload
from dct_schur.io import load_binary_plane, load_rgb
from evaluation.metrics import nc


def test_registry_has_one_proposal_and_all_baselines() -> None:
    specs = all_method_specs()
    proposals = [spec for spec in specs if spec.method_kind == "proposal"]
    baselines = [spec for spec in specs if spec.method_kind == "baseline"]
    assert [spec.method_id for spec in proposals] == [DCT_SCHUR_ID]
    assert len(baselines) == 16
    assert set(spec.method_id for spec in baselines) == set(ALL_BASELINE_IDS)
    assert len(resolve_methods("all")) == 17
    assert all(spec.method_kind == "baseline" for spec in resolve_methods("baselines"))


def test_every_baseline_clean_round_trip() -> None:
    host = load_rgb("data/host/lenna.bmp")
    source = load_binary_plane("data/watermark/wm.png", side=64)
    for method_id in ALL_BASELINE_IDS:
        spec = get_baseline_spec(method_id)
        payload = prepare_payload(source, 64 if spec.common_4096_payload else 16)
        marked, key = embed_baseline(method_id, host, payload, seed=2026, repeat=1)
        recovered = extract_baseline(
            marked,
            key,
            original_host=host if spec.requires_original_host else None,
        )
        # Some paper-guided methods are not exactly 1.0 clean by design.
        assert nc(payload, recovered) >= 0.99, method_id
