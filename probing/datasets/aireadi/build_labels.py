#!/usr/bin/env python3
"""Materialise person-level binary labels for AIREADI Condition probing.

Input  : ``clinical_data/condition_occurrence.csv``  (OMOP, one row per event)
Output : ``datasets/aireadi/data/person_labels.parquet`` (one row per person,
         ``len(conditions) + 2`` columns -- ``person_id`` plus a ``has_*`` flag
         per condition listed in ``configs/tasks/aireadi_condition.yaml``,
         plus an ``n_conditions`` summary column).

Usage::

    conda run -n torch2.5 python -m datasets.aireadi.build_labels \
        --condition-csv /path/to/condition_occurrence.csv \
        --task-config configs/tasks/aireadi_condition.yaml \
        --out datasets/aireadi/data/person_labels.parquet

Idempotent: re-running with the same inputs produces a byte-identical file.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _load_task_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "conditions" not in cfg:
        raise ValueError(f"Task config {path} missing 'conditions' section")
    return cfg


def _scan_persons(condition_csv: Path) -> Set[str]:
    """Return every person_id mentioned in ``condition_occurrence.csv``."""
    persons: Set[str] = set()
    with open(condition_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            persons.add(row["person_id"])
    return persons


def _person_condition_matrix(
    condition_csv: Path,
    concept_id_to_label: Dict[str, str],
    all_persons: Set[str],
) -> pd.DataFrame:
    """Build a person × condition binary matrix.

    A person is positive for a condition if **at least one** row in the OMOP
    table maps their ``person_id`` to that ``condition_concept_id``.  We
    deliberately ignore start dates / status -- mh_occurrence rows in AIREADI
    are essentially self-reported lifetime history.
    """
    columns = list(concept_id_to_label.values())
    rows = {pid: {c: 0 for c in columns} for pid in all_persons}

    with open(condition_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            cid = row["condition_concept_id"]
            label = concept_id_to_label.get(cid)
            if label is None:
                continue
            pid = row["person_id"]
            rows[pid][label] = 1

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "person_id"
    df = df.reset_index()
    df["person_id"] = df["person_id"].astype(str)
    df["n_conditions"] = df[columns].sum(axis=1).astype("int32")
    for col in columns:
        df[col] = df[col].astype("int8")
    df = df.sort_values("person_id").reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--condition-csv", type=Path, required=True,
        help="Path to AIREADI clinical_data/condition_occurrence.csv",
    )
    parser.add_argument(
        "--task-config", type=Path,
        default=PACKAGE_ROOT / "configs" / "tasks" / "aireadi_condition.yaml",
    )
    parser.add_argument(
        "--out", type=Path,
        default=PACKAGE_ROOT / "datasets" / "aireadi" / "data"
        / "person_labels.parquet",
    )
    args = parser.parse_args()

    if not args.condition_csv.exists():
        print(f"[ERROR] condition CSV not found: {args.condition_csv}",
              file=sys.stderr)
        sys.exit(1)

    cfg = _load_task_config(args.task_config)
    conditions: List[dict] = cfg["conditions"]
    concept_id_to_label = {
        str(c["concept_id"]): c["label"] for c in conditions
    }
    if len(concept_id_to_label) != len(conditions):
        raise ValueError("Duplicate concept_id detected in task config.")

    print(f"Building labels for {len(conditions)} conditions ...")
    print(f"  source : {args.condition_csv}")
    print(f"  config : {args.task_config}")

    persons = _scan_persons(args.condition_csv)
    print(f"  persons in OMOP: {len(persons)}")

    df = _person_condition_matrix(args.condition_csv, concept_id_to_label, persons)
    print(f"  output rows    : {len(df)}")
    print(f"  output columns : {list(df.columns)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")

    # Quick sanity prints: positive counts per column.
    print("\nPer-condition positive counts:")
    for c in conditions:
        n_pos = int(df[c["label"]].sum())
        flag = " " if n_pos == c.get("count", n_pos) else " *"
        print(f"  {c['label']:25s}  {n_pos:5d}{flag}")


if __name__ == "__main__":
    main()
