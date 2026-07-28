from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from dct_schur.config import SchurConfig
from dct_schur.provenance import embed_sequence, verify_sequence
from dct_schur.provenance.keys import load_private_key, load_public_key
from dct_schur.transport import DataKey


def _read_video(path: str | Path) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    width = height = 0
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        height, width = frame_bgr.shape[:2]
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError("video contains no readable frames")
    return frames, fps, (width, height)


def _write_video(path: str | Path, frames: list[np.ndarray], fps: float, size: tuple[int, int]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size
    )
    if not writer.isOpened():
        raise ValueError(f"cannot create video: {target}")
    for frame_rgb in frames:
        writer.write(cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()
    return target


def embed_video_provenance(
    input_video: str | Path,
    output_video: str | Path,
    private_key_path: str | Path,
    key_bundle_path: str | Path,
    *,
    frame_stride: int = 1,
    manifest_path: str | Path | None = None,
    config: SchurConfig | Mapping[str, Any] | None = None,
) -> Path:
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least one")
    frames, fps, size = _read_video(input_video)
    selected_indexes = list(range(0, len(frames), int(frame_stride)))
    selected_frames = [frames[index] for index in selected_indexes]
    manifest = Path(manifest_path).read_bytes() if manifest_path is not None else None
    marked_selected, keys, records = embed_sequence(
        selected_frames,
        private_key=load_private_key(private_key_path),
        manifest_bytes=manifest,
        config=config,
    )
    output_frames = list(frames)
    for index, marked in zip(selected_indexes, marked_selected):
        output_frames[index] = marked
    _write_video(output_video, output_frames, fps, size)
    bundle = {
        "input_video": str(input_video),
        "output_video": str(output_video),
        "fps": fps,
        "frame_stride": int(frame_stride),
        "selected_frame_indexes": selected_indexes,
        "keys": [key.to_dict() for key in keys],
        "records": [record.summary() for record in records],
    }
    target = Path(key_bundle_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return target


def verify_video_provenance(
    video_path: str | Path,
    key_bundle_path: str | Path,
    public_key_path: str | Path,
) -> dict[str, Any]:
    frames, _fps, _size = _read_video(video_path)
    bundle = json.loads(Path(key_bundle_path).read_text(encoding="utf-8"))
    indexes = [int(index) for index in bundle["selected_frame_indexes"]]
    selected = [frames[index] for index in indexes]
    keys = [DataKey.from_mapping(value) for value in bundle["keys"]]
    verification = verify_sequence(
        selected, keys, public_key=load_public_key(public_key_path)
    )
    return {
        "valid": verification.valid,
        "signature_valid": list(verification.signature_valid),
        "chain_valid": verification.chain_valid,
        "index_valid": verification.index_valid,
        "asset_id_valid": verification.asset_id_valid,
        "failures": list(verification.failures),
    }


__all__ = ["embed_video_provenance", "verify_video_provenance"]
