from __future__ import annotations

from pathlib import Path

from pipelines.benchmark import run_comparison_benchmark


def test_proposal_and_baseline_share_one_benchmark(tmp_path: Path) -> None:
    result = run_comparison_benchmark(
        "data/host/lenna.bmp",
        "data/watermark/wm.png",
        tmp_path / "comparison.json",
        methods="dct_schur,dct_dm_qim_chen2001_blind",
        output_summary_csv=tmp_path / "summary.csv",
        output_attack_csv=tmp_path / "attacks.csv",
        attack_suite="sanity",
        baseline_parameters_path="configs/baseline_parameters.json",
    )
    assert result["methods"] == [
        "dct_schur_invariant_relational",
        "dct_dm_qim_chen2001_blind",
    ]
    assert len(result["summaries"]) == 2
    assert all(row["successful_trials"] == 1 for row in result["summaries"])
    assert (tmp_path / "comparison.json").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "attacks.csv").exists()
