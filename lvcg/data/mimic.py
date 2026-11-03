"""
MIMIC-IV ECG data loading utilities.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch

try:
    import wfdb
except ImportError:
    wfdb = None

# Standard 12 leads order used in the project
STANDARD_12_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
]

def load_mimic_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """Load the MIMIC manifest (JSONL format)."""
    manifest = []
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                manifest.append(json.loads(line))
    return manifest

def load_wfdb_record(record_path: str, target_fs: Optional[int] = None) -> Tuple[np.ndarray, int, List[str]]:
    """Load a single WFDB record and return (data, fs, sig_names).
    
    Data shape: [num_leads, num_samples]
    """
    if wfdb is None:
        raise ImportError("wfdb package is required to load raw MIMIC data. Run: pip install wfdb")
    
    # record_path should not include extension
    record = wfdb.rdrecord(record_path)
    data = record.p_signal.T # [C, T]
    fs = record.fs
    sig_names = record.sig_name
    
    return data, fs, sig_names

def normalize_ecg(data: np.ndarray, scale_method: str = "minmax") -> np.ndarray:
    """
    Normalize ECG data using specified scaling method.
    
    Args:
        data: [num_leads, T] ECG data
        scale_method: Scaling method, one of:
            - "minmax": Per-lead min-max normalization to [0, 1]
            - "zscore": Per-lead z-score standardization (mean=0, std=1)
            - "zscore_scaled": z-score scaled to [-0.5, 0.5]
            - "global_minmax": Global min-max normalization to [0, 1]
            - "global_zscore": Global z-score standardization
            - "minmax_centered": min-max to [-0.5, 0.5]
            - "minmax_zeromean": baseline removal + min-max to [0, 1]
            - "zscore_01": z-score then linear mapping to [0, 1]
            - "none": No normalization
    
    Returns:
        Normalized data with same shape
    """
    if scale_method == "minmax":
        # Per-lead min-max to [0, 1]
        vmin = np.nanmin(data, axis=-1, keepdims=True)  # [12, 1]
        vmax = np.nanmax(data, axis=-1, keepdims=True)  # [12, 1]
        data = (data - vmin) / (vmax - vmin + 1e-8)
    
    elif scale_method == "zscore":
        # Per-lead z-score standardization
        mean = np.nanmean(data, axis=-1, keepdims=True)  # [12, 1]
        std = np.nanstd(data, axis=-1, keepdims=True)    # [12, 1]
        data = (data - mean) / (std + 1e-8)
    
    elif scale_method == "global_minmax":
        # Global min-max to [0, 1]
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        data = (data - vmin) / (vmax - vmin + 1e-8)
    
    elif scale_method == "global_zscore":
        # Global z-score standardization
        mean = np.nanmean(data)
        std = np.nanstd(data)
        data = (data - mean) / (std + 1e-8)
    
    elif scale_method == "minmax_centered":
        # Centered Min-Max: range [-0.5, 0.5], mean approx 0
        vmin = np.nanmin(data, axis=-1, keepdims=True)
        vmax = np.nanmax(data, axis=-1, keepdims=True)
        data = (data - vmin) / (vmax - vmin + 1e-8)
        data = data - 0.5
    
    elif scale_method == "zscore_scaled":
        # Scaled z-score: range approx [-1, 1]
        # Steps:
        # 1. z-score standardization (mean=0, std=1)
        # 2. Divide by 2, mapping +/-2 sigma to approx [-1, 1]
        # 3. Clip extreme values to [-1, 1]
        mean = np.nanmean(data, axis=-1, keepdims=True)
        std = np.nanstd(data, axis=-1, keepdims=True)
        data = (data - mean) / (std + 1e-8)  # z-score
        data = data / 2  # Scale, std approx 0.5
        data = np.clip(data, -1, 1)  # Clip extreme values
    
    elif scale_method == "minmax_zeromean":
        # Remove baseline first, then min-max to [0, 1]
        # Step 1: Remove baseline
        mean = np.nanmean(data, axis=-1, keepdims=True)
        data = data - mean
        
        # Step 2: min-max to [0, 1]
        vmin = np.nanmin(data, axis=-1, keepdims=True)
        vmax = np.nanmax(data, axis=-1, keepdims=True)
        data = (data - vmin) / (vmax - vmin + 1e-8)
    
    elif scale_method == "zscore_01":
        # z-score then linear mapping to [0, 1]
        # Steps:
        # 1. z-score standardization
        # 2. Clip and linear map: [-3, 3] -> [0, 1]
        mean = np.nanmean(data, axis=-1, keepdims=True)
        std = np.nanstd(data, axis=-1, keepdims=True)
        data = (data - mean) / (std + 1e-8)
        data = np.clip(data, -3, 3)
        data = (data + 3) / 6

    elif scale_method == "none":
        pass  # No normalization
    
    else:
        raise ValueError(f"Unknown scale_method: {scale_method}. "
                        f"Choose from: minmax, zscore, zscore_scaled, global_minmax, "
                        f"global_zscore, minmax_centered, minmax_zeromean, zscore_01, none")
    
    return data


def preprocess_mimic_record(
    data: np.ndarray, 
    fs_in: int, 
    sig_names: List[str],
    fs_out: int = 100,
    low: float = 0.67,
    high: float = 40.0,
    scale_method: str = "minmax"
) -> np.ndarray:
    """Preprocess raw ECG data: reorder leads, resample, bandpass filter, and normalize.
    
    Args:
        data: Raw ECG data [num_leads, T]
        fs_in: Input sampling frequency
        sig_names: List of signal names
        fs_out: Output sampling frequency (default: 100)
        low: Low cutoff frequency for bandpass filter (default: 0.67)
        high: High cutoff frequency for bandpass filter (default: 40.0)
        scale_method: Scaling method for normalization (default: "minmax")
            Options: "minmax", "zscore", "global_minmax", "global_zscore", "none"
    
    Returns:
        Preprocessed ECG data [12, T]
    """
    # 1. Reorder to standard 12 leads
    name_to_idx = {name.upper(): i for i, name in enumerate(sig_names)}
    selected = []
    for name in STANDARD_12_LEADS:
        # Try both upper and exact match
        idx = name_to_idx.get(name.upper())
        if idx is not None:
            selected.append(data[idx])
        else:
            # Pad with zeros if lead is missing
            selected.append(np.zeros_like(data[0]))
    data = np.stack(selected, axis=0) # [12, T]

    # 2. Resample
    if fs_in != fs_out:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(fs_in, fs_out)
        data = resample_poly(data, fs_out // g, fs_in // g, axis=-1)

    # 3. Bandpass filter
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs_out
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    data = filtfilt(b, a, data, axis=-1)

    # 4. Normalize using specified method
    data = normalize_ecg(data, scale_method=scale_method)
    
    # 5. Handle NaNs: fill with 0 after normalization
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return data.astype(np.float32)

class MimicLoader:
    """Convenience class to iterate over MIMIC records from a manifest."""
    
    def __init__(self, manifest_path: str, fs_out: int = 100, scale_method: str = "minmax"):
        """
        Args:
            manifest_path: Path to MIMIC manifest (JSONL format)
            fs_out: Output sampling frequency (default: 100)
            scale_method: Scaling method for normalization (default: "minmax")
                Options: "minmax", "zscore", "global_minmax", "global_zscore", "none"
        """
        self.manifest = load_mimic_manifest(manifest_path)
        self.fs_out = fs_out
        self.scale_method = scale_method

    def __len__(self) -> int:
        return len(self.manifest)

    def get_record(self, idx: int) -> Dict[str, Any]:
        item = self.manifest[idx]
        path = item['ecg_path']
        
        raw_data, fs_in, sig_names = load_wfdb_record(path)
        processed_data = preprocess_mimic_record(
            raw_data, fs_in, sig_names, 
            fs_out=self.fs_out, 
            scale_method=self.scale_method
        )
        
        # Extract label from assistant message if present
        label_text = ""
        for msg in item.get('messages', []):
            if msg.get('role') == 'assistant':
                label_text = msg.get('content', "")
                break
        
        return {
            "ecg": processed_data, # [12, T]
            "id": item['id'],
            "label_text": label_text,
            "fs": self.fs_out
        }

