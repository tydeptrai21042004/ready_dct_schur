from __future__ import annotations

from .config import SchurConfig
from .engine import embed_plane, extract_evidence, extract_evidence_candidates, extract_plane
from .key import SchurKey
from .philosophy import philosophy_summary

embed = embed_plane
extract = extract_plane

__all__ = [
    "SchurConfig", "SchurKey", "embed", "extract", "embed_plane",
    "extract_plane", "extract_evidence", "extract_evidence_candidates",
    "philosophy_summary",
]
