#!/usr/bin/env python3
"""Check that raw_root and vendored split CSVs are consistent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.datasets.benchmark_dataset import (  # noqa: E402
    resolve_raw_data_path,
    resolve_splits_dir,
)

BENCHMARKS = [
    "ptbxl_super_class",
    "ptbxl_sub_class",
    "ptbxl_form",
    "ptbxl_rhythm",
    "icbeb",
    "chapman",
]


def _check_ptbxl(raw_path: Path, split_dir: Path, name: str, max_check: int = 3) -> list[str]:
    errors = []
    train_csv = split_dir / f"{name}_train.csv"
    if not train_csv.exists():
        return [f"missing split file: {train_csv}"]
    df = pd.read_csv(train_csv, nrows=max_check)
    for rel in df["filename_hr"].head(max_check):
        full = raw_path / rel
        if not full.with_suffix(".hea").exists() and not full.exists():
            errors.append(f"PTB record not found: {full}")
    return errors


def _check_icbeb(raw_path: Path, split_dir: Path, name: str, max_check: int = 3) -> list[str]:
    errors = []
    train_csv = split_dir / f"{name}_train.csv"
    if not train_csv.exists():
        return [f"missing split file: {train_csv}"]
    df = pd.read_csv(train_csv, nrows=max_check)
    for fn in df["filename"].head(max_check):
        found = any(
            (raw_path / ts / f"{fn}.mat").exists()
            for ts in ("TrainingSet1", "TrainingSet2", "TrainingSet3")
        )
        if not found:
            errors.append(f"ICBEB .mat not found for {fn} under {raw_path}")
    return errors


def _check_chapman(raw_path: Path, split_dir: Path, name: str, max_check: int = 3) -> list[str]:
    errors = []
    train_csv = split_dir / f"{name}_train.csv"
    if not train_csv.exists():
        return [f"missing split file: {train_csv}"]
    df = pd.read_csv(train_csv, nrows=max_check)
    for p in df["ecg_path"].head(max_check):
        rel = p.replace("/chapman/", "")
        full = raw_path / rel
        if not full.exists() and not full.with_suffix(".mat").exists():
            errors.append(f"Chapman record not found: {full}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify LVCG probing data layout")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "probing" / "data_splits",
    )
    args = parser.parse_args()

    if not args.raw_root.is_dir():
        print(f"[ERROR] raw-root not found: {args.raw_root}", file=sys.stderr)
        sys.exit(1)

    all_errors: list[str] = []
    for name in BENCHMARKS:
        raw_sub = resolve_raw_data_path(args.raw_root, name)
        split_sub = resolve_splits_dir(args.splits_root, name)
        if not raw_sub.is_dir():
            all_errors.append(f"{name}: raw dir missing: {raw_sub}")
            continue
        if "ptbxl" in name:
            all_errors.extend(_check_ptbxl(raw_sub, split_sub, name))
        elif name == "icbeb":
            all_errors.extend(_check_icbeb(raw_sub, split_sub, name))
        else:
            all_errors.extend(_check_chapman(raw_sub, split_sub, name))

    if all_errors:
        print("Verification failed:")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)

    print("OK: raw_root and split CSVs look consistent (spot-checked first rows).")


if __name__ == "__main__":
    main()
