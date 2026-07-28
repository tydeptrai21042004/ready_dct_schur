from __future__ import annotations

from pathlib import Path

from pipelines.checkpointed import run_checkpointed_benchmark


def test_checkpointed_benchmark_resumes(tmp_path: Path) -> None:
    kwargs = dict(
        host_path="data/host/lenna.bmp",
        payload_path="data/watermark/wm.png",
        output_directory=tmp_path,
        methods="dct_dm_qim_chen2001_blind",
        attack_suite="clean",
        baseline_parameters_path="configs/baseline_parameters.json",
    )
    first = run_checkpointed_benchmark(**kwargs)
    assert first["status"] == "complete"
    checkpoint = tmp_path / "dct_dm_qim_chen2001_blind.json"
    before = checkpoint.stat().st_mtime_ns
    second = run_checkpointed_benchmark(**kwargs)
    assert second["status"] == "complete"
    assert checkpoint.stat().st_mtime_ns == before
