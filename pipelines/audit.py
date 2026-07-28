from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from baselines import ALL_BASELINE_IDS
from benchmarking.registry import DCT_SCHUR_ID, all_method_specs
from dct_schur.philosophy import philosophy_summary

# These are removed proposal implementation file/directory names, not baseline citations.
_REMOVED_PROPOSAL_PATH_TERMS = (
    "proposals/dct_qr",
    "proposals/cd_detqr",
    "spatial_cd_detqr",
    "direct_schur_rescue",
)


def audit_repository(root: str | Path, *, max_files_per_folder: int = 100) -> dict[str, Any]:
    base = Path(root)
    directory_counts: dict[str, int] = {}
    removed_proposal_paths: list[str] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        relative = str(path.relative_to(base)).replace("\\", "/").lower()
        directory = str(path.parent.relative_to(base))
        directory_counts[directory] = directory_counts.get(directory, 0) + 1
        if any(term in relative for term in _REMOVED_PROPOSAL_PATH_TERMS):
            removed_proposal_paths.append(relative)

    over_limit = {
        directory: count
        for directory, count in directory_counts.items()
        if count > int(max_files_per_folder)
    }
    specs = all_method_specs()
    proposal_ids = [spec.method_id for spec in specs if spec.method_kind == "proposal"]
    baseline_ids = [spec.method_id for spec in specs if spec.method_kind == "baseline"]
    one_proposal = proposal_ids == [DCT_SCHUR_ID]
    baselines_complete = set(baseline_ids) == set(ALL_BASELINE_IDS) and len(baseline_ids) == 16
    no_src_layout = not (base / "src").exists()
    passed = one_proposal and baselines_complete and no_src_layout and not removed_proposal_paths and not over_limit
    return {
        "single_proposal": one_proposal,
        "proposal_ids": proposal_ids,
        "baseline_count": len(baseline_ids),
        "baseline_ids": baseline_ids,
        "baselines_complete": baselines_complete,
        "removed_proposal_paths": removed_proposal_paths,
        "no_src_layout": no_src_layout,
        "max_files_per_folder": int(max_files_per_folder),
        "folders_over_limit": over_limit,
        "folder_file_counts": directory_counts,
        "philosophy": philosophy_summary(),
        "passed": passed,
    }


def write_audit(root: str | Path, output_path: str | Path) -> Path:
    result = audit_repository(root)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return target


__all__ = ["audit_repository", "write_audit"]
