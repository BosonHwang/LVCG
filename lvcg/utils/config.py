"""
Configuration loading and management.
"""

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class Config:
    """Structured config backed by a plain dict.

    Notes:
        - We intentionally use JSON to avoid non-stdlib dependencies (YAML parsers).
        - The config is logically separated into sections:
          { "model": {...}, "train": {...}, "data": {...}, "run": {...} }
    """
    raw: Dict[str, Any]

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def train(self) -> Dict[str, Any]:
        return self.raw.get("train", {})

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def run(self) -> Dict[str, Any]:
        return self.raw.get("run", {})


def load_config(path: str) -> Config:
    """Load a config file (YAML or JSON).

    Notes:
        - We prefer stdlib; however, YAML parsing requires PyYAML. If unavailable,
          raise a clear error. JSON remains fully stdlib.
        - Expect top-level sections: model / train / data / run.
    """
    if path.endswith((".yml", ".yaml")):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError("PyYAML is required for YAML configs. Install with `pip install pyyaml`.") from e
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    return Config(raw=raw)


def add_cli_overrides(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add minimal CLI overrides for convenience.

    We allow changing the config path and a few common scalar overrides without
    introducing arbitrary nesting or complex parsing.
    """
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config.")
    parser.add_argument("--run.m", type=str, default=None, help="RunId code for 'm' slot (e.g., m1).")
    parser.add_argument("--run.s", type=str, default=None, help="RunId code for 's' slot (e.g., s1).")
    parser.add_argument("--run.k", type=str, default=None, help="RunId code for 'k' slot (e.g., k1).")
    parser.add_argument("--train.batch_size", type=int, default=None)
    parser.add_argument("--train.max_steps", type=int, default=None)
    parser.add_argument("--data.meta_root", type=str, default=None)
    return parser


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply minimal CLI overrides to the loaded config."""
    raw = dict(cfg.raw)  # shallow copy is fine for our simple usage

    def set_if_not_none(path_keys: Tuple[str, str], value: Optional[Any]) -> None:
        if value is None:
            return
        section, key = path_keys
        if section not in raw:
            raw[section] = {}
        raw[section][key] = value

    set_if_not_none(("run", "m"), getattr(args, "run.m"))
    set_if_not_none(("run", "s"), getattr(args, "run.s"))
    set_if_not_none(("run", "k"), getattr(args, "run.k"))
    set_if_not_none(("train", "batch_size"), getattr(args, "train.batch_size"))
    set_if_not_none(("train", "max_steps"), getattr(args, "train.max_steps"))
    set_if_not_none(("data", "meta_root"), getattr(args, "data.meta_root"))
    return Config(raw=raw)

