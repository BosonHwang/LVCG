#!/usr/bin/env python3
"""Materialise person-level train/val/test splits for AIREADI Condition probing.

We default to the dataset authors' own ``recommended_split`` column from
``participants.tsv`` (train 1576 / val 352 / test 352) because:

  * It already enforces person-level grouping (no person leaks across splits).
  * It is stratified by ``study_group`` (healthy / pre_diabetes / oral_med /
    insulin), which is the dominant confounder in AIREADI.
  * Using it keeps results comparable to other AIREADI benchmarks.

The output also stores the ``study_group`` so downstream code can do
sub-cohort analysis without re-reading the participants table.

Output: ``datasets/aireadi/data/splits/v1.csv``
        person_id,split,study_group,age,clinical_site,recommended_split
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Set

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _read_participants(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        # Strip BOM from the very first column header if present.
        for row in r:
            normalised = {
                (k.lstrip("\ufeff") if isinstance(k, str) else k): v
                for k, v in row.items()
            }
            rows.append(normalised)
    return rows


def _read_manifest_persons(manifest_tsv: Path) -> Set[str]:
    """Return person_ids that have at least one ECG recording."""
    persons: Set[str] = set()
    with open(manifest_tsv) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            persons.add(row["person_id"])
    return persons


def _read_label_persons(parquet_path: Path) -> Set[str]:
    if not parquet_path.exists():
        return set()
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=["person_id"])
    return set(df["person_id"].astype(str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--aireadi-root", type=Path, required=True,
        help="Root of the AIREADI pilot release (contains cardiac_ecg/)",
    )
    parser.add_argument(
        "--labels-parquet", type=Path,
        default=PACKAGE_ROOT / "datasets" / "aireadi" / "data"
        / "person_labels.parquet",
        help="If present, restrict the split to persons that also have labels.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=PACKAGE_ROOT / "datasets" / "aireadi" / "data" / "splits" / "v1.csv",
    )
    parser.add_argument(
        "--strategy", choices=["recommended"], default="recommended",
        help="Split source. Currently only the dataset authors' "
             "'recommended_split' column is supported.",
    )
    args = parser.parse_args()

    participants_tsv = args.aireadi_root / "participants.tsv"
    manifest_tsv = args.aireadi_root / "cardiac_ecg" / "manifest.tsv"
    if not participants_tsv.exists():
        print(f"[ERROR] {participants_tsv} not found", file=sys.stderr)
        sys.exit(1)
    if not manifest_tsv.exists():
        print(f"[ERROR] {manifest_tsv} not found", file=sys.stderr)
        sys.exit(1)

    rows = _read_participants(participants_tsv)
    ecg_persons = _read_manifest_persons(manifest_tsv)
    label_persons = _read_label_persons(args.labels_parquet)
    have_labels = bool(label_persons)

    keep = []
    counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    dropped_no_ecg = 0
    dropped_no_label = 0
    dropped_no_split = 0

    for row in rows:
        pid = row["person_id"]
        if pid not in ecg_persons:
            dropped_no_ecg += 1
            continue
        if have_labels and pid not in label_persons:
            dropped_no_label += 1
            continue
        split = row.get("recommended_split", "").strip()
        if split not in counts:
            dropped_no_split += 1
            continue
        keep.append({
            "person_id": pid,
            "split": split,
            "study_group": row.get("study_group", "").strip(),
            "age": row.get("age", "").strip(),
            "clinical_site": row.get("clinical_site", "").strip(),
            "recommended_split": split,
        })
        counts[split] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["person_id", "split", "study_group", "age",
                        "clinical_site", "recommended_split"],
        )
        w.writeheader()
        for r in keep:
            w.writerow(r)

    total = sum(counts.values())
    print(f"Wrote {args.out}  ({total} persons)")
    print(f"  train: {counts['train']}  val: {counts['val']}  test: {counts['test']}")
    print(f"  dropped (no ECG):     {dropped_no_ecg}")
    if have_labels:
        print(f"  dropped (no labels):  {dropped_no_label}")
    print(f"  dropped (no split):   {dropped_no_split}")

    # Per-split study_group breakdown for sanity.
    from collections import Counter
    for sp in ("train", "val", "test"):
        c = Counter(r["study_group"] for r in keep if r["split"] == sp)
        print(f"  {sp:5s} study_group: {dict(c)}")


if __name__ == "__main__":
    main()
