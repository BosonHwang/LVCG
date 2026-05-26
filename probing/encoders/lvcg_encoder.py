"""LVCG encoder wrapper for linear probing."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lvcg.models.lvcg import LVCG

from .base import BaseEncoder


class LVCGEncoder(BaseEncoder):
    """Frozen LVCG backbone for embedding extraction.

    Operates at 100 Hz with ``time_len=1000`` (10 s). Output is 640-d
    ``ecg_emb`` (struct 256 + dynamic 256 + rhythm 128).
    """

    TARGET_FS = 100
    TARGET_LEN = 1000

    def __init__(self, checkpoint: str):
        super().__init__()

        self.backbone = LVCG(
            time_len=self.TARGET_LEN,
            num_leads=12,
            lead_order="mimic",
            beat_len=128,
            state_dim=256,
            max_beats=20,
            beat_encoder_stem=48,
            beat_encoder_stages=[96, 192, 256, 256],
            beat_encoder_blocks=[3, 3, 3, 2],
            beat_encoder_kernel=7,
            beat_encoder_dropout=0.1,
            gru_hidden_dim=256,
            gru_num_layers=2,
            gru_dropout=0.1,
            decoder_initial_channels=128,
            decoder_hidden_channels=[128, 64, 64, 32, 32],
            rr_num_basis=8,
            fs=self.TARGET_FS,
            use_ttt=False,
            temporal_type="gru",
        )

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.eval()

        self.out_features = self.backbone.out_features

    def _resample(self, ecg: torch.Tensor, source_fs: int = 500) -> torch.Tensor:
        if source_fs != self.TARGET_FS:
            new_len = int(ecg.shape[-1] * self.TARGET_FS / source_fs)
            ecg = F.interpolate(ecg, size=new_len, mode="linear", align_corners=False)
        T = ecg.shape[-1]
        if T > self.TARGET_LEN:
            ecg = ecg[..., : self.TARGET_LEN]
        elif T < self.TARGET_LEN:
            ecg = F.pad(ecg, (0, self.TARGET_LEN - T))
        return ecg

    @torch.no_grad()
    def ext_ecg_emb(self, ecg: torch.Tensor, source_fs: int = 500) -> torch.Tensor:
        ecg = self._resample(ecg, source_fs)
        return self.backbone.ext_ecg_emb(ecg)
