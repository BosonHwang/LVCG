"""Abstract base class for frozen-backbone encoders used by the probing pipeline."""

from abc import abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    """Frozen ECG encoder with a uniform ``ext_ecg_emb`` API.

    Subclasses must set ``out_features`` and implement ``ext_ecg_emb``.
    """

    out_features: int

    @torch.no_grad()
    @abstractmethod
    def ext_ecg_emb(self, ecg: torch.Tensor) -> torch.Tensor:
        """Extract a fixed-size embedding from raw 12-lead ECG.

        Args:
            ecg: ``[B, 12, T]`` raw ECG tensor at 500 Hz (MELP convention).
                 Each encoder handles its own resampling internally.

        Returns:
            Tensor of shape ``[B, out_features]``.
        """
        raise NotImplementedError
