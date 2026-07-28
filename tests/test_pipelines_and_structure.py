from __future__ import annotations

from pathlib import Path

from pipelines.audit import audit_repository
from pipelines.data import embed_text_value, extract_text_value
from pipelines.image import embed_image, extract_image


def test_file_pipelines(tmp_path: Path) -> None:
    image_result = embed_image(
        "data/host/lenna.bmp",
        "data/watermark/wm.png",
        tmp_path / "marked.png",
        tmp_path / "image_key.json",
    )
    assert image_result.psnr_db > 50.0
    assert image_result.clean_nc == 1.0
    assert extract_image(
        image_result.output_image, image_result.key_file, tmp_path / "payload.png"
    ).exists()

    data_result = embed_text_value(
        "data/host/lenna.bmp",
        "single-proposal benchmark repository",
        tmp_path / "data_marked.png",
        tmp_path / "data_key.json",
    )
    assert data_result.psnr_db > 50.0
    assert extract_text_value(data_result.output_image, data_result.key_file) == "single-proposal benchmark repository"


def test_repository_has_single_proposal_baselines_and_no_src_layout() -> None:
    root = Path(".")
    assert not (root / "src").exists()
    audit = audit_repository(root)
    assert audit["single_proposal"]
    assert not audit["folders_over_limit"]
