"""WFDB-backed PyTorch dataset for AIREADI 12-lead ECGs.

The AIREADI release stores each ECG as a (.hea, .dat) pair under
``cardiac_ecg/ecg_12lead/<vendor>/<person_id>/...``.  Sampling rate is 500 Hz
across the cohort (Philips PageWriter TC30); recording length varies but is
typically ~11 s.  We resample/crop/pad to a fixed ``target_len`` (default
5000 samples = 10 s @ 500 Hz) so the output tensor matches every probing
encoder's ``(B, 12, 5000)`` contract.

Lead order in the WFDB headers matches the standard 12-lead order
(``I, II, III, aVR, aVL, aVF, V1..V6``); we trust it without re-mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset

# Lead order expected by the probing encoders (MELP convention).
_EXPECTED_LEADS: Sequence[str] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)


def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-lead z-score normalisation (matches MELP's default)."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def _crop_or_pad(x: np.ndarray, target_len: int) -> np.ndarray:
    """Centre-crop or right-pad with zeros to ``target_len`` samples."""
    n = x.shape[-1]
    if n == target_len:
        return x
    if n > target_len:
        start = (n - target_len) // 2
        return x[..., start:start + target_len]
    pad = np.zeros((x.shape[0], target_len - n), dtype=x.dtype)
    return np.concatenate([x, pad], axis=-1)


def _reorder_leads(signal: np.ndarray, sig_names: Sequence[str]) -> np.ndarray:
    """Reorder ``signal`` so its leads follow :data:`_EXPECTED_LEADS`."""
    if list(sig_names) == list(_EXPECTED_LEADS):
        return signal
    name_to_idx = {n: i for i, n in enumerate(sig_names)}
    missing = [n for n in _EXPECTED_LEADS if n not in name_to_idx]
    if missing:
        raise ValueError(
            f"AIREADI record missing expected leads {missing}; got {list(sig_names)}"
        )
    indices = [name_to_idx[n] for n in _EXPECTED_LEADS]
    return signal[indices]


class AIREADIECGDataset(Dataset):
    """Person-indexed ECG dataset.

    Args:
        person_ids: Persons to include (must all have an entry in ``manifest``).
        manifest:   Mapping ``person_id -> absolute path stem`` (no .hea/.dat).
        labels:     ``(N, num_classes)`` float tensor aligned with ``person_ids``.
        target_len: Number of samples per lead (default 5000 = 10 s @ 500 Hz).
        norm_method: ``"zscore"`` or ``"none"``.
        transform:  Optional callable ``(np.ndarray) -> np.ndarray`` applied
                    after normalisation but before tensor conversion.
    """

    def __init__(
        self,
        person_ids: List[str],
        manifest: dict,
        labels: torch.Tensor,
        target_len: int = 5000,
        norm_method: str = "zscore",
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        if len(person_ids) != labels.shape[0]:
            raise ValueError(
                f"person_ids ({len(person_ids)}) and labels "
                f"({labels.shape[0]}) length mismatch"
            )
        missing = [p for p in person_ids if p not in manifest]
        if missing:
            raise ValueError(
                f"{len(missing)} person_ids missing from manifest "
                f"(first few: {missing[:5]})"
            )
        self.person_ids = list(person_ids)
        self.manifest = manifest
        self.labels = labels
        self.target_len = target_len
        self.norm_method = norm_method
        self.transform = transform

    # The probing pipeline reads ``num_classes`` directly from the dataset.
    @property
    def num_classes(self) -> int:
        return self.labels.shape[1]

    def __len__(self) -> int:
        return len(self.person_ids)

    def _load_ecg(self, person_id: str) -> np.ndarray:
        stem = self.manifest[person_id]
        rec = wfdb.rdrecord(str(stem))
        sig = rec.p_signal.T.astype(np.float32)  # (12, T)
        sig = _reorder_leads(sig, rec.sig_name)
        sig = _crop_or_pad(sig, self.target_len)
        if self.norm_method == "zscore":
            sig = _zscore(sig)
        elif self.norm_method != "none":
            raise ValueError(f"Unknown norm_method: {self.norm_method}")
        if self.transform is not None:
            sig = self.transform(sig)
        return sig

    def __getitem__(self, idx: int) -> dict:
        pid = self.person_ids[idx]
        ecg = torch.from_numpy(self._load_ecg(pid))
        return {
            "ecg": ecg,
            "label": self.labels[idx],
            "person_id": pid,
        }
