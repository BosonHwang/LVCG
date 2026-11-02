"""
Data loading pipeline for ECG datasets.

Provides:
- ECGDataset: Load pre-processed .npy files
- MimicRawDataset: Load raw MIMIC-IV WFDB files with train/val split
- make_dataloader: Convenience function to create DataLoaders
"""

from dataclasses import dataclass
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


from .mimic import MimicLoader


class ECGDataset(Dataset):
    """Dataset for pre-processed ECG .npy files."""
    
    def __init__(self, root: str, split: str, num_leads: int = 12, time_len: Optional[int] = None):
        super().__init__()
        self.root = root
        self.split = split
        self.num_leads = num_leads
        self.time_len = time_len

        split_dir = os.path.join(root, split)
        self.files: List[str] = []
        if os.path.isdir(split_dir):
            for fn in sorted(os.listdir(split_dir)):
                if fn.endswith(".npy"):
                    self.files.append(os.path.join(split_dir, fn))

        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {split_dir}. Dataset cannot be empty.")
        
        self.length = len(self.files)

    def __len__(self) -> int:
        return self.length

    def _load_file(self, path: str) -> Dict[str, Any]:
        arr = np.load(path, allow_pickle=True).item()
        ecg = arr["ecg"]  # expected shape [L, T]
        assert ecg.ndim == 2, "ECG must be [L, T]"
        assert ecg.shape[0] == self.num_leads, "Lead count mismatch"
        if self.time_len is not None:
            T_src = ecg.shape[1]
            if T_src > self.time_len:
                # Center crop
                start = (T_src - self.time_len) // 2
                ecg = ecg[:, start:start + self.time_len]
            elif T_src < self.time_len:
                raise ValueError(f"Data too short: got {T_src}, need {self.time_len} (no padding)")
        label = arr.get("label", None)
        id_str = arr.get("id", os.path.basename(path))
        return {"ecg": ecg.astype(np.float32), "label": label, "id": id_str}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        return self._load_file(path)


class MimicRawDataset(Dataset):
    """
    Dataset for raw MIMIC-IV WFDB files using a manifest.
    
    Supports train/val split via indices parameter.
    """
    
    def __init__(
        self, 
        manifest_path: str, 
        fs_out: int = 100, 
        time_len: Optional[int] = None, 
        num_leads: int = 12,
        indices: Optional[List[int]] = None,
        scale_method: str = "minmax"
    ):
        """
        Args:
            manifest_path: Path to JSONL manifest file
            fs_out: Target sampling rate
            time_len: Target time length (crop if longer, error if shorter)
            num_leads: Expected number of leads
            indices: Optional list of indices to use (for train/val split)
            scale_method: Scaling method for normalization (default: "minmax")
        """
        super().__init__()
        self.loader = MimicLoader(manifest_path, fs_out=fs_out, scale_method=scale_method)
        self.time_len = time_len
        self.num_leads = num_leads
        
        # If indices provided, use them; otherwise use all
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(len(self.loader)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        actual_idx = self.indices[idx]
        item = self.loader.get_record(actual_idx)
        ecg = item["ecg"]
        
        if self.time_len is not None:
            T_src = ecg.shape[1]
            if T_src > self.time_len:
                # Center crop
                start = (T_src - self.time_len) // 2
                ecg = ecg[:, start:start + self.time_len]
            elif T_src < self.time_len:
                raise ValueError(f"Data too short: got {T_src}, need {self.time_len} (no padding)")
            
        return {
            "ecg": ecg.astype(np.float32),
            "label": None,
            "id": item["id"]
        }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for ECG batches."""
    ecgs = [torch.from_numpy(x["ecg"]) if isinstance(x["ecg"], np.ndarray) else x["ecg"] for x in batch]
    labels = [x["label"] for x in batch]
    ids = [x["id"] for x in batch]

    T_list = [e.shape[1] for e in ecgs]
    assert len(set(T_list)) == 1, "All items must share the same T"
    L_list = [e.shape[0] for e in ecgs]
    assert len(set(L_list)) == 1, "All items must share the same lead count"

    ecg_tensor = torch.stack(ecgs, dim=0)  # [B, L, T]
    label_tensor = None
    if any(lbl is not None for lbl in labels):
        label_vals = [(-1 if lbl is None else int(lbl)) for lbl in labels]
        label_tensor = torch.tensor(label_vals, dtype=torch.long)
    return {"ecg": ecg_tensor, "label": label_tensor, "id": ids}


def split_indices(
    total_size: int, 
    train_ratio: float = 0.9, 
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Split indices into train and validation sets.
    
    Args:
        total_size: Total number of samples
        train_ratio: Ratio for training set (default 0.9)
        seed: Random seed for reproducibility
    
    Returns:
        (train_indices, val_indices)
    """
    indices = list(range(total_size))
    
    # Shuffle with fixed seed for reproducibility
    rng = random.Random(seed)
    rng.shuffle(indices)
    
    # Split
    split_point = int(total_size * train_ratio)
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    return train_indices, val_indices


def make_dataloader(cfg: Dict[str, Any], split: str) -> DataLoader:
    """
    Construct DataLoader for the given split.

    Config keys (data section):
        - dataset_type: 'npy' (default) or 'mimic_raw'
        - meta_root: base directory for .npy or path to manifest for mimic_raw
        - fs: target sampling rate (default 100)
        - num_leads: expected lead count (default 12)
        - time_len: target time length (crop if longer, error if shorter)
        - train_ratio: train/val split ratio (default 0.9)
        - split_seed: random seed for split (default 42)
        - scale_method: normalization method (default 'minmax')
    """
    data_cfg = cfg.get("data", {})
    dtype = data_cfg.get("dataset_type", "npy")
    root = data_cfg.get("meta_root", "./data")
    fs = int(data_cfg.get("fs", 100))
    num_leads = int(data_cfg.get("num_leads", 12))
    time_len = data_cfg.get("time_len", None)
    if time_len is not None:
        time_len = int(time_len)
    
    train_ratio = float(data_cfg.get("train_ratio", 0.9))
    split_seed = int(data_cfg.get("split_seed", 42))
    scale_method = str(data_cfg.get("scale_method", "minmax"))
    
    batch_size = int(cfg.get("train", {}).get("batch_size", 4))
    num_workers = int(cfg.get("train", {}).get("num_workers", 0))

    if dtype == "mimic_raw":
        # For mimic_raw, first get total size, then split
        loader = MimicLoader(root, fs_out=fs, scale_method=scale_method)
        total_size = len(loader)
        
        train_indices, val_indices = split_indices(total_size, train_ratio, split_seed)
        
        if split == "train":
            indices = train_indices
            shuffle = True
        else:  # val
            indices = val_indices
            shuffle = False
        
        ds = MimicRawDataset(
            manifest_path=root, 
            fs_out=fs, 
            time_len=time_len, 
            num_leads=num_leads,
            indices=indices,
            scale_method=scale_method
        )
    else:
        ds = ECGDataset(root=root, split=split, num_leads=num_leads, time_len=time_len)
        shuffle = (split == "train")
        
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        collate_fn=_collate
    )


def make_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Create both train and validation DataLoaders.
    
    Args:
        cfg: Full config dictionary
    
    Returns:
        (train_loader, val_loader)
    """
    train_loader = make_dataloader(cfg, split="train")
    val_loader = make_dataloader(cfg, split="val")
    return train_loader, val_loader

