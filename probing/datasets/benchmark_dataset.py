"""Native PyTorch datasets for PTB-XL, ICBEB (CPSC 2018), and Chapman (CSN).

Split CSVs live under ``probing/data_splits/`` (vendored from the MELP
benchmark splits for reproducibility). Raw WFDB/MAT files are read from a
user-provided ``raw_root``; see ``docs/DATA_LAYOUT.md``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.io import loadmat
from torch.utils.data import Dataset

# Standard 12-lead order expected by LVCG probing encoders.
_EXPECTED_NUM_LEADS = 12
_TARGET_LEN = 5000


def _normalize_ecg(ecg: np.ndarray, norm_method: str) -> np.ndarray:
    if norm_method == "minmax":
        return (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg) + 1e-8)
    if norm_method == "zscore":
        mean = ecg.mean(axis=-1, keepdims=True)
        std = ecg.std(axis=-1, keepdims=True)
        return (ecg - mean) / (std + 1e-8)
    raise ValueError(f"Unknown norm_method: {norm_method}")


def _swap_avl_avf(ecg: np.ndarray) -> np.ndarray:
    """Match MIMIC lead order: swap aVL and aVF (indices 4 and 5)."""
    out = ecg.copy()
    out[[4, 5]] = out[[5, 4]]
    return out


class BenchmarkECGDataset(Dataset):
    """Single-split ECG dataset for linear probing benchmarks."""

    def __init__(
        self,
        data_path: str,
        csv_file: pd.DataFrame,
        split: str,
        dataset_name: str,
        data_pct: float = 1.0,
        norm_method: str = "zscore",
        subsample_seed: int = 42,
    ) -> None:
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.split = split
        self.norm_method = norm_method

        if split == "train" and data_pct < 1.0:
            csv_file = (
                csv_file.sample(frac=data_pct, random_state=subsample_seed)
                .reset_index(drop=True)
            )
        else:
            csv_file = csv_file.reset_index(drop=True)

        if "ptbxl" in dataset_name:
            self.labels_name = list(csv_file.columns[6:])
            self.num_classes = len(self.labels_name)
            self.ecg_path = csv_file["filename_hr"]
            self.labels = csv_file.iloc[:, 6:].values
            self.patient_id = csv_file["patient_id"]
            self.ecg_id = csv_file["ecg_id"]

        elif dataset_name == "icbeb":
            self.labels_name = list(csv_file.columns[7:])
            self.num_classes = len(self.labels_name)
            ecg_paths: List[str] = []
            for filename in csv_file["filename"]:
                found = False
                for ts in ("TrainingSet1", "TrainingSet2", "TrainingSet3"):
                    mat_path = os.path.join(data_path, ts, f"{filename}.mat")
                    if os.path.exists(mat_path):
                        ecg_paths.append(os.path.join(ts, f"{filename}.mat"))
                        found = True
                        break
                if not found:
                    ecg_paths.append(f"TrainingSet1/{filename}.mat")
            self.ecg_path = pd.Series(ecg_paths)
            self.labels = csv_file.iloc[:, 7:].values
            self.patient_id = csv_file["patient_id"]
            self.ecg_id = csv_file["ecg_id"]

        elif dataset_name == "chapman":
            self.labels_name = list(csv_file.columns[3:])
            self.num_classes = len(self.labels_name)
            csv_file = csv_file.copy()
            csv_file["ecg_path"] = csv_file["ecg_path"].apply(
                lambda x: x.replace("/chapman/", "")
            )
            exists = csv_file["ecg_path"].apply(
                lambda x: os.path.exists(os.path.join(data_path, x))
            )
            csv_file = csv_file[exists.values].reset_index(drop=True)
            self.ecg_path = csv_file["ecg_path"]
            self.labels = csv_file.iloc[:, 3:].values
            self.ecg_id = csv_file["ecg_path"].apply(
                lambda x: x.split("/")[-1].split(".")[0]
            )
            self.patient_id = csv_file["ecg_path"].apply(
                lambda x: x.split("/")[-2]
            )
        else:
            raise ValueError(
                f"Unsupported dataset_name {dataset_name!r}; "
                "expected ptbxl_*, icbeb, or chapman"
            )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict:
        if "ptbxl" in self.dataset_name:
            ecg_path = os.path.join(self.data_path, self.ecg_path.iloc[idx])
            ecg = wfdb.rdsamp(ecg_path)[0].T.astype(np.float32)
            ecg = ecg[:, :_TARGET_LEN]
            ecg_id = self.ecg_id.iloc[idx]
            patient_id = self.patient_id.iloc[idx]

        elif self.dataset_name == "icbeb":
            ecg_path = os.path.join(self.data_path, self.ecg_path.iloc[idx])
            mat_data = loadmat(ecg_path)
            ecg = mat_data["ECG"]["data"][0, 0].astype(np.float32)
            ecg = ecg[:, :2500]
            ecg = np.pad(ecg, ((0, 0), (0, 2500)), mode="constant")
            ecg = ecg[:, :_TARGET_LEN]
            ecg_id = self.ecg_id.iloc[idx]
            patient_id = self.patient_id.iloc[idx]

        else:  # chapman
            ecg_path = os.path.join(self.data_path, self.ecg_path.iloc[idx])
            ecg = loadmat(ecg_path)["val"].astype(np.float32)
            ecg = ecg[:, :_TARGET_LEN]
            ecg_id = self.ecg_id.iloc[idx]
            patient_id = self.patient_id.iloc[idx]

        if ecg.shape[0] != _EXPECTED_NUM_LEADS:
            raise ValueError(
                f"Expected {_EXPECTED_NUM_LEADS} leads, got {ecg.shape[0]} "
                f"for {ecg_path}"
            )

        ecg = _normalize_ecg(ecg, self.norm_method)
        ecg = _swap_avl_avf(ecg)

        target = torch.from_numpy(self.labels[idx].astype(np.float32))
        uid = f"{patient_id}_{ecg_id}"

        return {
            "id": uid,
            "patient_id": patient_id,
            "ecg": torch.from_numpy(ecg).float(),
            "label": target,
        }


def resolve_raw_data_path(raw_root: Path, dataset_name: str) -> Path:
    if "ptbxl" in dataset_name:
        return raw_root / "ptbxl"
    if dataset_name == "icbeb":
        return raw_root / "icbeb"
    if dataset_name == "chapman":
        return raw_root / "csn"
    raise ValueError(f"Unknown benchmark dataset: {dataset_name}")


def resolve_splits_dir(splits_root: Path, dataset_name: str) -> Path:
    if "ptbxl" in dataset_name:
        task = dataset_name.replace("ptbxl_", "")
        return splits_root / "ptbxl" / task
    if dataset_name == "icbeb":
        return splits_root / "icbeb"
    if dataset_name == "chapman":
        return splits_root / "chapman"
    raise ValueError(f"Unknown benchmark dataset: {dataset_name}")
