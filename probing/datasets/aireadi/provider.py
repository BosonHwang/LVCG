"""AIREADI Condition probing :class:`DatasetProvider`.

Wires together:

* the offline ``person_labels.parquet`` (built by :mod:`build_labels`),
* the offline ``splits/v1.csv``        (built by :mod:`make_split`),
* the WFDB ECG manifest                (shipped with the AIREADI release),
* the task-group definition in ``configs/tasks/aireadi_condition.yaml``.

A single instance can serve any of the configured ``task_group`` keys
(``primary_10`` / ``composite_3`` / ``all_30``) without rebuilding offline
products.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from torch.utils.data import DataLoader

from ..base import DatasetBundle, DatasetProvider
from .ecg_dataset import AIREADIECGDataset


@dataclass
class _TaskSpec:
    """Resolved task-group spec: ordered list of label columns or composites."""

    name: str
    columns: List[str]                                  # for primary / all_30
    composites: List[Dict[str, List[str]]]              # for composite_3
    label_names: List[str]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _read_split_csv(path: Path) -> Dict[str, List[str]]:
    """Return ``{'train': [pid, ...], 'val': [...], 'test': [...]}``."""
    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            split = row["split"]
            if split not in out:
                continue
            out[split].append(str(row["person_id"]))
    return out


def _build_manifest(manifest_tsv: Path, ecg_root: Path) -> Dict[str, str]:
    """Map ``person_id`` -> WFDB record stem (no extension).

    AIREADI's manifest stores POSIX-absolute paths from the dataset root
    (e.g. ``/cardiac_ecg/...``); we strip the leading slash and join with the
    user-supplied ``ecg_root``.  When a person has multiple ECGs we keep the
    first one in manifest order.
    """
    manifest: Dict[str, str] = {}
    with open(manifest_tsv) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            pid = row["person_id"]
            if pid in manifest:
                continue
            hea = row["wfdb_hea_filepath"].lstrip("/")
            stem = str(ecg_root / hea).removesuffix(".hea")
            manifest[pid] = stem
    return manifest


def _resolve_task(cfg: dict, group: str) -> _TaskSpec:
    """Validate ``group`` against the task-config and return a normalised spec."""
    groups = cfg.get("task_groups", {})
    if group not in groups:
        raise ValueError(
            f"Unknown task_group '{group}'. Available: {list(groups)}"
        )
    spec = groups[group]
    all_cond_labels = [c["label"] for c in cfg["conditions"]]

    if "composites" in spec:
        composites = []
        for entry in spec["composites"]:
            members = entry["members"]
            unknown = [m for m in members if m not in all_cond_labels]
            if unknown:
                raise ValueError(
                    f"Composite '{entry['name']}' references unknown columns: "
                    f"{unknown}"
                )
            composites.append({"name": entry["name"], "members": list(members)})
        return _TaskSpec(
            name=group,
            columns=[],
            composites=composites,
            label_names=[c["name"] for c in composites],
        )

    cols = spec.get("columns")
    if cols == "ALL" or cols is None:
        cols = all_cond_labels
    unknown = [c for c in cols if c not in all_cond_labels]
    if unknown:
        raise ValueError(
            f"task_group '{group}' references unknown columns: {unknown}"
        )
    return _TaskSpec(
        name=group,
        columns=list(cols),
        composites=[],
        label_names=list(cols),
    )


def _materialise_labels(
    df, person_ids: List[str], task: _TaskSpec
) -> torch.Tensor:
    """Build an ``(N, num_classes)`` float tensor for ``person_ids``."""
    sub = df.set_index("person_id").loc[person_ids]
    if task.composites:
        cols = []
        for entry in task.composites:
            mat = sub[entry["members"]].values.astype("int8")
            cols.append((mat.sum(axis=1) > 0).astype("float32"))
        arr = torch.from_numpy(_stack_columns(cols))
    else:
        arr = torch.from_numpy(
            sub[task.columns].values.astype("float32")
        )
    return arr


def _stack_columns(cols):
    import numpy as np
    return np.stack(cols, axis=1)


def _subsample_ratio(
    person_ids: List[str], ratio: float, seed: int = 42
) -> List[str]:
    """Deterministic per-ratio training subsample (matches MELP semantics)."""
    if ratio >= 1.0:
        return person_ids
    if ratio <= 0:
        raise ValueError(f"label_ratio must be > 0, got {ratio}")
    n = max(1, int(round(len(person_ids) * ratio)))
    rng = random.Random(seed)
    pool = list(person_ids)
    rng.shuffle(pool)
    return sorted(pool[:n])


# ── Provider ────────────────────────────────────────────────────────────────


class AIREADIConditionProvider(DatasetProvider):
    """ECG + multi-label classification on the AIREADI Condition cohort.

    Args:
        ecg_root:          Filesystem root that contains ``cardiac_ecg/``.
        manifest_tsv:      Path to ``cardiac_ecg/manifest.tsv``.
        labels_parquet:    Output of :mod:`build_labels`.
        split_csv:         Output of :mod:`make_split`.
        task_config:       Path to ``configs/tasks/aireadi_condition.yaml``.
        task_group:        One of ``primary_10`` / ``composite_3`` / ``all_30``.
        target_len:        ECG sample count (default 5000 = 10 s @ 500 Hz).
        norm_method:       ``"zscore"`` or ``"none"``.
        subsample_seed:    Seed for the deterministic ``label_ratio`` subsample.
    """

    def __init__(
        self,
        ecg_root: str,
        manifest_tsv: str,
        labels_parquet: str,
        split_csv: str,
        task_config: str,
        task_group: str = "primary_10",
        target_len: int = 5000,
        norm_method: str = "zscore",
        subsample_seed: int = 42,
    ) -> None:
        self.ecg_root = Path(ecg_root)
        self.manifest_tsv = Path(manifest_tsv)
        self.labels_parquet = Path(labels_parquet)
        self.split_csv = Path(split_csv)
        self.task_config_path = Path(task_config)
        self.task_group = task_group
        self.target_len = target_len
        self.norm_method = norm_method
        self.subsample_seed = subsample_seed

    def _check_paths(self) -> None:
        for label, p in [
            ("ecg_root", self.ecg_root),
            ("manifest_tsv", self.manifest_tsv),
            ("labels_parquet", self.labels_parquet),
            ("split_csv", self.split_csv),
            ("task_config", self.task_config_path),
        ]:
            if not p.exists():
                raise FileNotFoundError(
                    f"AIREADI provider: {label} not found at {p}.\n"
                    f"Did you run datasets/aireadi/build_labels.py and "
                    f"datasets/aireadi/make_split.py?"
                )

    def _load_task_config(self) -> dict:
        with open(self.task_config_path) as f:
            return yaml.safe_load(f)

    def build(
        self,
        label_ratio: float,
        batch_size: int,
        num_workers: int,
    ) -> DatasetBundle:
        self._check_paths()

        import pandas as pd

        cfg = self._load_task_config()
        task = _resolve_task(cfg, self.task_group)

        df = pd.read_parquet(self.labels_parquet)
        df["person_id"] = df["person_id"].astype(str)

        splits = _read_split_csv(self.split_csv)
        manifest = _build_manifest(self.manifest_tsv, self.ecg_root)

        # Restrict to persons that have (a) labels and (b) ECG available.
        labelled = set(df["person_id"])
        for sp in splits:
            splits[sp] = [
                p for p in splits[sp]
                if p in labelled and p in manifest
            ]

        # Apply label_ratio to the training split only (val/test stay full).
        train_pids = _subsample_ratio(
            splits["train"], label_ratio, seed=self.subsample_seed
        )
        val_pids = splits["val"]
        test_pids = splits["test"]

        train_y = _materialise_labels(df, train_pids, task)
        val_y = _materialise_labels(df, val_pids, task)
        test_y = _materialise_labels(df, test_pids, task)

        def _make(ds_pids, ys, shuffle):
            ds = AIREADIECGDataset(
                person_ids=ds_pids,
                manifest=manifest,
                labels=ys,
                target_len=self.target_len,
                norm_method=self.norm_method,
            )
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )

        train_loader = _make(train_pids, train_y, shuffle=True)
        val_loader = _make(val_pids, val_y, shuffle=False)
        test_loader = _make(test_pids, test_y, shuffle=False)

        meta = {
            "source": "aireadi_condition",
            "task_group": self.task_group,
            "n_train_persons": len(train_pids),
            "n_val_persons": len(val_pids),
            "n_test_persons": len(test_pids),
            "label_positive_counts": {
                name: int(train_y[:, i].sum().item())
                for i, name in enumerate(task.label_names)
            },
        }

        return DatasetBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=len(task.label_names),
            label_names=task.label_names,
            meta=meta,
        )


__all__ = ["AIREADIConditionProvider"]
