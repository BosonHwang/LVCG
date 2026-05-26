"""
Classification heads for downstream tasks.
"""

from typing import Optional

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Simple classification head on top of W [B, D].

    Supports:
        - 'linear': single linear probe
        - 'mlp': small MLP for slightly more capacity
    """

    def __init__(self, kind: str, in_dim: int, num_classes: int, hidden: int = 128):
        super().__init__()
        kind = kind.lower()
        if kind == "linear":
            self.net = nn.Linear(in_dim, num_classes)
        elif kind == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, num_classes),
            )
        else:
            raise ValueError("kind must be 'linear' or 'mlp'")

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return self.net(W)

