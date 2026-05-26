#!/usr/bin/env python3
"""Run 7-dataset linear probing evaluation for a trained LVCG checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBING_ROOT = REPO_ROOT / "probing"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval" / "probing.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="LVCG linear probing evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Probing config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override LVCG checkpoint path in config",
    )
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results",
        type=str,
        default="probing/results/probing_results.csv",
        help="Output CSV (relative to repo root or absolute)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.checkpoint is not None:
        for model_cfg in cfg.get("models", {}).values():
            if model_cfg.get("type") == "lvcg":
                model_cfg["checkpoint"] = args.checkpoint

    cmd = [
        sys.executable,
        str(PROBING_ROOT / "run_probing.py"),
        "--config",
        str(config_path),
        "--models",
        "lvcg",
        "--ratio",
        str(args.ratio),
        "--seed",
        str(args.seed),
        "--results",
        str(args.results),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(PROBING_ROOT))


if __name__ == "__main__":
    main()
