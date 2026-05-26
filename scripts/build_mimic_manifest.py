#!/usr/bin/env python3
"""Build a JSONL manifest for MIMIC-IV ECG pretraining.

Each line is one JSON object with:
  - id: unique string id
  - ecg_path: absolute WFDB record path without .hea/.dat suffix
  - messages: optional list with user/assistant roles (assistant may be empty)

Example::

    python scripts/build_mimic_manifest.py \\
        --mimic-root /path/to/mimic-iv-ecg/files \\
        --out /path/to/mimic_manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_wfdb_stems(root: Path):
    for hea in sorted(root.rglob("*.hea")):
        if hea.name.startswith("."):
            continue
        yield hea.with_suffix("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MIMIC-IV ECG JSONL manifest")
    parser.add_argument(
        "--mimic-root",
        type=Path,
        required=True,
        help="Root directory containing MIMIC WFDB records (e.g. .../files/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL manifest path",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="mimic",
        help="Record id prefix (default: mimic)",
    )
    args = parser.parse_args()

    if not args.mimic_root.is_dir():
        raise SystemExit(f"mimic-root not found: {args.mimic_root}")

    stems = list(_iter_wfdb_stems(args.mimic_root))
    if not stems:
        raise SystemExit(f"No .hea files under {args.mimic_root}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, stem in enumerate(stems):
            row = {
                "id": f"{args.prefix}_{i}",
                "ecg_path": str(stem.resolve()),
                "messages": [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": ""},
                ],
            }
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(stems)} records to {args.out}")


if __name__ == "__main__":
    main()
