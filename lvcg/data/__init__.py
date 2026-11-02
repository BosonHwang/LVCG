"""LVCG data processing modules.

The legacy ``src.data.pipeline`` module (``ECGDataset`` / ``MimicRawDataset`` /
``make_dataloader``) is no longer shipped in this checkout but downstream
``src.models.lvcg_modeling`` only needs ``angle`` and ``BeatSegmenter``.  We
import ``pipeline`` lazily so encoder-only code paths (e.g. probing /
fine-tuning that injects its own dataloader) still work even when the legacy
loaders are absent.
"""

from .angle import get_lead_directions, get_lead_angles, reorder_leads
from .beat_segmentation import BeatSegmenter

try:  # legacy training-time loaders; safe to skip when missing
    from .pipeline import (
        ECGDataset,
        MimicRawDataset,
        make_dataloader,
        make_dataloaders,
    )
    _HAS_PIPELINE = True
except ImportError:  # pragma: no cover - pipeline removed in this checkout
    ECGDataset = None  # type: ignore[assignment]
    MimicRawDataset = None  # type: ignore[assignment]
    make_dataloader = None  # type: ignore[assignment]
    make_dataloaders = None  # type: ignore[assignment]
    _HAS_PIPELINE = False

__all__ = [
    "ECGDataset",
    "MimicRawDataset",
    "make_dataloader",
    "make_dataloaders",
    "BeatSegmenter",
    "get_lead_directions",
    "get_lead_angles",
    "reorder_leads",
]

