"""LVCG-native provider for PTB-XL / ICBEB / Chapman probing benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .base import DatasetBundle, DatasetProvider
from .benchmark_dataset import (
    BenchmarkECGDataset,
    resolve_raw_data_path,
    resolve_splits_dir,
)


class BenchmarkProvider(DatasetProvider):
    """Loads standard ECG benchmarks using vendored split CSVs + user raw_root."""

    def __init__(
        self,
        dataset_name: str,
        raw_root: str,
        splits_root: Optional[str] = None,
        norm_method: str = "zscore",
        chapman_classes: Optional[List[str]] = None,
        subsample_seed: int = 42,
    ) -> None:
        self.dataset_name = dataset_name
        self.raw_root = Path(raw_root)
        repo_root = Path(__file__).resolve().parents[2]
        self.splits_root = (
            Path(splits_root) if splits_root else repo_root / "data_splits"
        )
        self.norm_method = norm_method
        self.chapman_classes = chapman_classes
        self.subsample_seed = subsample_seed

    def build(
        self,
        label_ratio: float,
        batch_size: int,
        num_workers: int,
    ) -> DatasetBundle:
        data_path = resolve_raw_data_path(self.raw_root, self.dataset_name)
        split_dir = resolve_splits_dir(self.splits_root, self.dataset_name)

        train_csv = pd.read_csv(split_dir / f"{self.dataset_name}_train.csv")
        val_csv = pd.read_csv(split_dir / f"{self.dataset_name}_val.csv")
        test_csv = pd.read_csv(split_dir / f"{self.dataset_name}_test.csv")

        train_ds = BenchmarkECGDataset(
            data_path=str(data_path),
            csv_file=train_csv,
            split="train",
            dataset_name=self.dataset_name,
            data_pct=label_ratio,
            norm_method=self.norm_method,
            subsample_seed=self.subsample_seed,
        )
        val_ds = BenchmarkECGDataset(
            data_path=str(data_path),
            csv_file=val_csv,
            split="val",
            dataset_name=self.dataset_name,
            data_pct=1.0,
            norm_method=self.norm_method,
        )
        test_ds = BenchmarkECGDataset(
            data_path=str(data_path),
            csv_file=test_csv,
            split="test",
            dataset_name=self.dataset_name,
            data_pct=1.0,
            norm_method=self.norm_method,
        )

        if self.dataset_name == "chapman" and self.chapman_classes:
            train_loader, val_loader, test_loader, num_classes, label_names = (
                _filter_chapman_loaders(
                    train_ds, val_ds, test_ds,
                    self.chapman_classes, batch_size, num_workers,
                )
            )
        else:
            train_loader = _make_loader(train_ds, batch_size, num_workers, shuffle=True)
            val_loader = _make_loader(val_ds, batch_size, num_workers, shuffle=False)
            test_loader = _make_loader(test_ds, batch_size, num_workers, shuffle=False)
            num_classes = train_ds.num_classes
            label_names = list(train_ds.labels_name)

        return DatasetBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            label_names=label_names,
            meta={
                "source": "benchmark",
                "dataset_name": self.dataset_name,
                "raw_root": str(self.raw_root),
                "splits_root": str(self.splits_root),
            },
        )


def _make_loader(ds, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )


def _filter_chapman_loaders(
    train_ds: BenchmarkECGDataset,
    val_ds: BenchmarkECGDataset,
    test_ds: BenchmarkECGDataset,
    target_classes: List[str],
    batch_size: int,
    num_workers: int,
):
    """Restrict Chapman to a label subset (paper uses SR, AFIB, 1AVB, RBBB)."""
    for ds in (train_ds, val_ds, test_ds):
        names = list(ds.labels_name)
        indices = []
        for cls in target_classes:
            if cls not in names:
                raise ValueError(
                    f"Chapman class {cls!r} not in split columns: {names}"
                )
            indices.append(names.index(cls))
        labels = np.asarray(ds.labels)
        ds.labels = labels[:, indices]
        ds.labels_name = list(target_classes)
        ds.num_classes = len(target_classes)

    train_loader = _make_loader(train_ds, batch_size, num_workers, shuffle=True)
    val_loader = _make_loader(val_ds, batch_size, num_workers, shuffle=False)
    test_loader = _make_loader(test_ds, batch_size, num_workers, shuffle=False)
    return train_loader, val_loader, test_loader, len(target_classes), list(target_classes)
