from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb(path: str | Path, image: np.ndarray) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(target)
    return target


def load_binary_plane(path: str | Path, side: int = 64) -> np.ndarray:
    image = Image.open(path).convert("L").resize((side, side), Image.Resampling.NEAREST)
    return (np.asarray(image, dtype=np.uint8) > 127).astype(np.uint8) * 255


def save_binary_plane(path: str | Path, plane: np.ndarray) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(plane) > 0).astype(np.uint8) * 255, mode="L").save(target)
    return target


__all__ = ["load_rgb", "save_rgb", "load_binary_plane", "save_binary_plane"]
