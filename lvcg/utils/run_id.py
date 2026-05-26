"""
Run ID management utilities.
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class RunIdSpec:
    """Builds compact run identifiers (m#s#k#) and maps codes to human-readable tags."""
    mapping: Dict[str, Dict[str, str]]
    codes: Tuple[str, str, str]

    def build(self) -> str:
        m, s, k = self.codes
        return f"{m}{s}{k}"


def ensure_run_dirs(base_dir: str, run_id: str) -> str:
    """Create base_dir/run_id and return the path."""
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def step_dir(run_dir: str, step: int) -> str:
    """Return/create run_dir/step_{step}/."""
    sd = os.path.join(run_dir, f"step_{step}")
    os.makedirs(sd, exist_ok=True)
    return sd

