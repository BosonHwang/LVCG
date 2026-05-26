"""
Beat Modules for LVCG.

Contains all beat-level processing modules:
- BeatEncoder (ResNet1D)
- ContextMixer (Transformer)
- BeatDecoder
- BeatStitcher
- RREmbedding
- GlobalRREmbedding
- DynamicsHead (GRU)
- EmbeddingReadout
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# BeatEncoder (ResNet1D)
# =============================================================================

class ResidualBlock1D(nn.Module):
    """
    Residual block for 1D signal.
    
    Structure: x -> Conv -> BN -> GELU -> Conv -> BN -> (+x) -> GELU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L] -> [B, C', L']"""
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.act(out)
        
        return out


class ResNet1DStage(nn.Module):
    """A stage of ResNet with multiple residual blocks."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        kernel_size: int = 7,
        stride: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        blocks = [
            ResidualBlock1D(in_channels, out_channels, kernel_size, stride, dropout)
        ]
        for _ in range(1, num_blocks):
            blocks.append(
                ResidualBlock1D(out_channels, out_channels, kernel_size, 1, dropout)
            )
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class BeatEncoder(nn.Module):
    """
    ResNet1D encoder for VCG beats.
    
    Input: [B, N, 3, P=128] - N beats, each is 3D VCG
    Output: [B, N, D=256] - Beat-level state vectors
    
    Key Constraint: Only sees single beat (no cross-beat interaction)
    """
    
    def __init__(
        self,
        beat_len: int = 128,
        state_dim: int = 256,
        stem_channels: int = 64,
        stage_channels: List[int] = [128, 256, 512, 512],
        stage_blocks: List[int] = [4, 6, 6, 3],
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        
        # Stem: 3 -> stem_channels
        self.stem = nn.Sequential(
            nn.Conv1d(3, stem_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
        )
        
        # Build stages
        stages = []
        in_ch = stem_channels
        for out_ch, n_blocks in zip(stage_channels, stage_blocks):
            stages.append(
                ResNet1DStage(in_ch, out_ch, n_blocks, kernel_size, stride=2, dropout=dropout)
            )
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection
        self.proj = nn.Linear(stage_channels[-1], state_dim)
        
        self.out_features = state_dim
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, beats: torch.Tensor) -> torch.Tensor:
        """
        Encode VCG beats to state vectors.
        
        Args:
            beats: [B, N, 3, P] - N beats, each is 3-channel VCG
        
        Returns:
            z: [B, N, D] - Beat-level state vectors
        """
        B, N, C, P = beats.shape
        assert C == 3, f"Expected 3 VCG channels, got {C}"
        
        # Reshape: process all beats independently
        x = beats.view(B * N, C, P)  # [B*N, 3, P]
        
        # Stem
        x = self.stem(x)  # [B*N, 64, P/2]
        
        # Stages
        x = self.stages(x)  # [B*N, 512, ~P/32]
        
        # Global pooling
        x = self.pool(x).squeeze(-1)  # [B*N, 512]
        
        # Project
        z = self.proj(x)  # [B*N, D]
        
        # Reshape back
        z = z.view(B, N, self.state_dim)  # [B, N, D]
        
        return z


# =============================================================================
# ContextMixer (Transformer)
# =============================================================================

class ContextMixer(nn.Module):
    """
    Transformer encoder for cross-beat context aggregation.
    
    Input: z_fused [B, N, D] - Beat states fused with RR embedding
    Output: z_ctx [B, N, D] - Contextualized beat states
    
    WARNING: Do NOT use z_ctx for temporal prediction (info leakage)
             Only use for Readout/MoCo.
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_beats: int = 20,
    ):
        super().__init__()
        self.state_dim = state_dim
        
        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_beats, state_dim) * 0.02)
        
        # Layer norm before transformer
        self.pre_norm = nn.LayerNorm(state_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=state_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final layer norm
        self.post_norm = nn.LayerNorm(state_dim)
        
        self.out_features = state_dim
    
    def forward(
        self,
        z_fused: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Contextualize beat states.
        
        Args:
            z_fused: [B, N, D] - Beat states fused with RR
            mask: [B, N] - Valid beat mask (True = valid)
        
        Returns:
            z_ctx: [B, N, D] - Contextualized states for readout
        """
        B, N, D = z_fused.shape
        
        # Add positional embedding
        x = z_fused + self.pos_embed[:, :N, :]
        
        # Pre-norm
        x = self.pre_norm(x)
        
        # Create attention mask for padding
        if mask is not None:
            src_key_padding_mask = ~mask  # True = ignore
        else:
            src_key_padding_mask = None
        
        # Transformer forward
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Post-norm
        z_ctx = self.post_norm(x)
        
        return z_ctx


# =============================================================================
# BeatDecoder
# =============================================================================

class DecoderResBlock(nn.Module):
    """Residual block with upsampling for decoder."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        upsample: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.upsample = upsample
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels or upsample:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.act(out)
        
        return out


class BeatDecoder(nn.Module):
    """
    ResNet-style decoder from beat state to VCG waveform.
    
    Input: z [B, N, D=256] - Beat state vectors (without RR!)
    Output: V_hat_beats [B, N, 3, P=128] - Decoded VCG beats
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        beat_len: int = 128,
        initial_channels: int = 256,
        hidden_channels: List[int] = [256, 128, 128, 64, 64],
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.beat_len = beat_len
        self.initial_len = 4
        
        # Initial projection
        self.proj = nn.Linear(state_dim, initial_channels * self.initial_len)
        self.proj_bn = nn.BatchNorm1d(initial_channels)
        
        # Build decoder blocks
        blocks = []
        in_ch = initial_channels
        for out_ch in hidden_channels:
            blocks.append(DecoderResBlock(in_ch, out_ch, kernel_size, upsample=True, dropout=dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        
        # Final output layer
        self.out_conv = nn.Sequential(
            nn.Conv1d(hidden_channels[-1], hidden_channels[-1], kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_channels[-1]),
            nn.GELU(),
            nn.Conv1d(hidden_channels[-1], 3, 1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode state vectors to VCG beats.
        
        Args:
            z: [B, N, D] - Beat state vectors
        
        Returns:
            V_hat_beats: [B, N, 3, P] - Decoded VCG beats
        """
        B, N, D = z.shape
        
        z_flat = z.view(B * N, D)
        
        # Initial projection
        x = self.proj(z_flat)
        x = x.view(B * N, -1, self.initial_len)
        x = self.proj_bn(x)
        
        # Decoder blocks
        x = self.blocks(x)
        
        # Ensure exact output length
        if x.shape[-1] != self.beat_len:
            x = F.interpolate(x, size=self.beat_len, mode='linear', align_corners=False)
        
        # Output VCG
        x = self.out_conv(x)
        
        V_hat_beats = x.view(B, N, 3, self.beat_len)
        
        return V_hat_beats


# =============================================================================
# BeatStitcher
# =============================================================================

class BeatStitcher(nn.Module):
    """
    Stitch decoded beats back to continuous waveform using RR intervals.
    
    Input: V_hat_beats [B, N, 3, P] + rr_intervals [B, N]
    Output: V_hat [B, 3, T]
    """
    
    def __init__(
        self,
        beat_len: int = 128,
        target_len: int = 1024,
        crossfade_ratio: float = 0.1,
    ):
        super().__init__()
        self.beat_len = beat_len
        self.target_len = target_len
        self.crossfade_ratio = crossfade_ratio
    
    def forward(
        self,
        V_hat_beats: torch.Tensor,
        rr_intervals: torch.Tensor,
        beat_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stitch beats to continuous waveform.
        
        Args:
            V_hat_beats: [B, N, 3, P] - Decoded beats
            rr_intervals: [B, N] - Original RR lengths (samples)
            beat_mask: [B, N] - Valid beat mask
        
        Returns:
            V_hat: [B, 3, T] - Reconstructed continuous VCG
        """
        B, N, C, P = V_hat_beats.shape
        device = V_hat_beats.device
        dtype = V_hat_beats.dtype
        
        V_hat = torch.zeros((B, C, self.target_len), dtype=dtype, device=device)
        weight = torch.zeros((B, 1, self.target_len), dtype=dtype, device=device)
        
        for b in range(B):
            pos = 0
            
            # Count valid beats for this sample
            valid_indices = [n for n in range(N) if beat_mask[b, n]]
            num_valid = len(valid_indices)
            
            for beat_idx, n in enumerate(valid_indices):
                rr_len = int(rr_intervals[b, n].item())
                if rr_len <= 0:
                    continue
                
                # Resample beat from P to original RR length
                beat = V_hat_beats[b, n].unsqueeze(0)
                beat_resampled = F.interpolate(
                    beat, size=rr_len, mode='linear', align_corners=False
                ).squeeze(0)
                
                # Cross-fade window (but not at boundaries)
                fade_len = max(1, int(rr_len * self.crossfade_ratio))
                win = torch.ones(rr_len, dtype=dtype, device=device)
                
                # Only fade at start if not first beat
                if beat_idx > 0:
                    win[:fade_len] = torch.linspace(0, 1, fade_len, dtype=dtype, device=device)
                
                # Only fade at end if not last beat
                if beat_idx < num_valid - 1:
                    win[-fade_len:] = torch.linspace(1, 0, fade_len, dtype=dtype, device=device)
                
                # Determine output range
                end = min(pos + rr_len, self.target_len)
                actual_len = end - pos
                
                if actual_len <= 0:
                    break
                
                # Overlap-add
                V_hat[b, :, pos:end] += beat_resampled[:, :actual_len] * win[:actual_len]
                weight[b, 0, pos:end] += win[:actual_len]
                
                pos = end
        
        # Normalize
        V_hat = V_hat / weight.clamp(min=1e-6)
        
        return V_hat


# =============================================================================
# RREmbedding
# =============================================================================

class RREmbedding(nn.Module):
    """
    RR interval embedding with log normalization and sin/cos basis.
    
    Input: rr [B, N] - RR intervals
    Output: rr_emb [B, N, D] - RR embeddings
    
    Design: rr_norm = log(RR / mean(RR)), then sin/cos expansion
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_basis: int = 8,
        max_log_rr: float = 2.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_basis = num_basis
        self.max_log_rr = max_log_rr
        
        # Log-spaced frequencies
        freqs = torch.exp(torch.linspace(math.log(0.5), math.log(10.0), num_basis))
        self.register_buffer('freqs', freqs)
        
        # Projection
        self.proj = nn.Linear(2 * num_basis, embed_dim)
        
        self.out_features = embed_dim
    
    def forward(
        self,
        rr: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode RR intervals.
        
        Args:
            rr: [B, N] - RR intervals in samples
            mask: [B, N] - Valid beat mask
        
        Returns:
            rr_emb: [B, N, embed_dim]
        """
        B, N = rr.shape
        
        # Compute mean RR
        if mask is not None:
            rr_masked = rr.masked_fill(~mask, 0.0)
            rr_sum = rr_masked.sum(dim=1, keepdim=True)
            rr_count = mask.sum(dim=1, keepdim=True).clamp(min=1)
            rr_mean = rr_sum / rr_count
        else:
            rr_mean = rr.mean(dim=1, keepdim=True)
        
        # Normalize: log(RR / mean(RR))
        rr_norm = torch.log(rr / rr_mean.clamp(min=1e-6) + 1e-6)
        rr_norm = rr_norm.clamp(-self.max_log_rr, self.max_log_rr)
        
        # Sin/Cos basis expansion
        rr_expanded = rr_norm.unsqueeze(-1)  # [B, N, 1]
        freqs = self.freqs.view(1, 1, -1)  # [1, 1, num_basis]
        
        angles = 2 * math.pi * freqs * rr_expanded
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        
        rr_feat = torch.cat([sin_enc, cos_enc], dim=-1)
        
        # Project
        rr_emb = self.proj(rr_feat)
        
        return rr_emb


# =============================================================================
# GlobalRREmbedding
# =============================================================================

class GlobalRREmbedding(nn.Module):
    """
    Encode N RR intervals into a fixed-dimension global representation.
    
    Design:
    1. Statistical features: mean, std, range, CV (captures global heart rate and HRV)
    2. Sequence features: normalized + sin/cos basis (amplifies relative differences)
    3. Difference features: RR change trends
    4. Attention pooling: aggregate N features to fixed dimension
    
    Input: rr [B, N], mask [B, N]
    Output: rr_emb [B, out_dim]
    """
    
    def __init__(
        self,
        out_dim: int = 128,
        max_beats: int = 20,
        num_basis: int = 8,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_basis = num_basis
        self.hidden_dim = hidden_dim
        
        # Statistical features branch: 4 statistics -> hidden_dim
        self.stat_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Sequence features branch: each RR encoded with sin/cos basis
        self.seq_proj = nn.Linear(2 * num_basis, hidden_dim)
        
        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(max_beats, hidden_dim) * 0.02)
        
        # Attention pooling: [N, hidden_dim] -> [1, hidden_dim]
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Difference features: 2 dims (mean_diff, std_diff) -> hidden_dim
        self.diff_proj = nn.Linear(2, hidden_dim)
        
        # Final fusion: 3 * hidden_dim -> out_dim
        self.final_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        
        self.out_features = out_dim
    
    def forward(
        self,
        rr: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rr: [B, N] - RR intervals in samples
            mask: [B, N] - Valid beat mask (bool)
        
        Returns:
            rr_emb: [B, out_dim] - Global RR representation
        """
        B, N = rr.shape
        device = rr.device
        
        # Ensure mask is float for computation
        mask_float = mask.float()
        
        # === 1. Statistical features ===
        rr_masked = rr * mask_float
        count = mask_float.sum(dim=1, keepdim=True).clamp(min=1)
        mean_rr = rr_masked.sum(dim=1, keepdim=True) / count  # [B, 1]
        
        # Variance
        diff_sq = ((rr - mean_rr) ** 2) * mask_float
        var_rr = diff_sq.sum(dim=1) / count.squeeze(-1).clamp(min=1)
        std_rr = var_rr.sqrt()  # [B]
        
        # Min/Max (fill invalid positions with large/small values)
        rr_for_max = rr.masked_fill(~mask, -1e9)
        rr_for_min = rr.masked_fill(~mask, 1e9)
        max_rr = rr_for_max.max(dim=1).values  # [B]
        min_rr = rr_for_min.min(dim=1).values  # [B]
        
        # Statistical feature vector
        stats = torch.stack([
            mean_rr.squeeze(-1),                          # Mean RR
            std_rr,                                       # RR std (HRV)
            max_rr - min_rr,                              # RR range
            std_rr / mean_rr.squeeze(-1).clamp(min=1),    # Coefficient of variation
        ], dim=1)  # [B, 4]
        
        stat_emb = self.stat_proj(stats)  # [B, hidden_dim]
        
        # === 2. Sequence features (amplify differences) ===
        # Normalize: log(RR / mean_RR) amplifies relative differences
        rr_norm = torch.log(rr / mean_rr.clamp(min=1) + 1e-6)  # [B, N]
        rr_norm = rr_norm.clamp(-2, 2)  # Limit range
        
        # Sin/Cos basis expansion
        freqs = torch.arange(1, self.num_basis + 1, device=device).float()
        angles = rr_norm.unsqueeze(-1) * freqs * math.pi  # [B, N, num_basis]
        rr_feat = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B, N, 2*num_basis]
        
        seq_emb = self.seq_proj(rr_feat)  # [B, N, hidden_dim]
        seq_emb = seq_emb + self.pos_enc[:N]  # Add positional encoding
        
        # Attention pooling
        query = self.query.expand(B, -1, -1)  # [B, 1, hidden_dim]
        # key_padding_mask: True means ignore that position
        seq_pooled, _ = self.attn_pool(
            query, seq_emb, seq_emb,
            key_padding_mask=~mask,
        )
        seq_pooled = seq_pooled.squeeze(1)  # [B, hidden_dim]
        
        # === 3. Difference features ===
        rr_diff = rr[:, 1:] - rr[:, :-1]  # [B, N-1]
        diff_mask = mask[:, 1:] & mask[:, :-1]  # Both ends must be valid
        diff_mask_float = diff_mask.float()
        diff_count = diff_mask_float.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Relative difference (relative to mean_rr)
        rr_diff_norm = rr_diff / mean_rr.clamp(min=1)
        
        # Difference statistics
        diff_mean = (rr_diff_norm * diff_mask_float).sum(dim=1, keepdim=True) / diff_count
        diff_var = (((rr_diff_norm - diff_mean) ** 2) * diff_mask_float).sum(dim=1) / diff_count.squeeze(-1)
        diff_std = diff_var.sqrt()
        
        diff_stats = torch.stack([diff_mean.squeeze(-1), diff_std], dim=1)  # [B, 2]
        diff_emb = self.diff_proj(diff_stats)  # [B, hidden_dim]
        
        # === 4. Fusion ===
        combined = torch.cat([stat_emb, seq_pooled, diff_emb], dim=-1)  # [B, 3*hidden_dim]
        rr_emb = self.final_proj(combined)  # [B, out_dim]
        
        return rr_emb


# =============================================================================
# DynamicsHead (GRU)
# =============================================================================

class DynamicsHead(nn.Module):
    """
    Multi-layer causal GRU for temporal difference prediction.
    
    Input: z [B, N, D] + rr_emb [B, N, D]
    Output: delta_z_hat [B, N-k, D], delta_z_target [B, N-k, D], stride
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_stride: int = 3,
        use_rr: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.max_stride = max_stride
        self.use_rr = use_rr
        
        input_dim = state_dim * 2 if use_rr else state_dim
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        
        self.proj = nn.Linear(hidden_dim, state_dim)
    
    def forward(
        self,
        z: torch.Tensor,
        rr_emb: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Predict beat state differences.
        
        Args:
            z: [B, N, D] - Beat state vectors
            rr_emb: [B, N, D] - RR embedding
            stride: Prediction stride k
            mask: [B, N] - Valid beat mask
        
        Returns:
            delta_z_hat: [B, N-k, D] - Predicted differences
            delta_z_target: [B, N-k, D] - Ground truth differences (detached)
            stride: Actual stride used
        """
        import random
        
        B, N, D = z.shape
        
        # Sample stride
        if stride is None:
            max_k = min(self.max_stride, N - 1)
            stride = random.randint(1, max(1, max_k))
        stride = min(stride, N - 1)
        
        # Prepare GRU input
        if self.use_rr and rr_emb is not None:
            gru_input = torch.cat([z, rr_emb], dim=-1)
        else:
            gru_input = z
        
        # GRU forward
        gru_out, _ = self.gru(gru_input)
        
        # Project
        pred_all = self.proj(gru_out)
        
        # Select predictions
        delta_z_hat = pred_all[:, :N-stride, :]
        
        # Ground truth difference (detached)
        delta_z_target = (z[:, stride:, :] - z[:, :N-stride, :]).detach()
        
        return delta_z_hat, delta_z_target, stride


# =============================================================================
# EmbeddingReadout
# =============================================================================

class EmbeddingReadout(nn.Module):
    """
    Readout ECG embedding from contextualized beat states (z_ctx).
    
    Input: z_ctx [B, N, D] + optional delta_z [B, N-1, D]
    Output: ecg_emb [B, out_dim]
    
    Modes:
    - 'mean_std': concat(mean(z), std(z)) -> [B, 2D]
    - 'mean_std_delta': above + delta stats -> [B, 4D]
    - 'full': above + max, min -> [B, 6D]
    """
    
    MODES = ['mean_std', 'mean_std_delta', 'full']
    
    def __init__(
        self,
        state_dim: int = 256,
        mode: str = 'mean_std_delta',
    ):
        super().__init__()
        assert mode in self.MODES
        
        self.state_dim = state_dim
        self.mode = mode
        
        if mode == 'mean_std':
            self._out_features = state_dim * 2
        elif mode == 'mean_std_delta':
            self._out_features = state_dim * 4
        elif mode == 'full':
            self._out_features = state_dim * 6
    
    @property
    def out_features(self) -> int:
        return self._out_features
    
    def forward(
        self,
        z: torch.Tensor,
        delta_z: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract ECG embedding.
        
        Args:
            z: [B, N, D] - Beat state vectors (z_ctx)
            delta_z: [B, N-1, D] - State differences
            mask: [B, N] - Valid beat mask
        
        Returns:
            ecg_emb: [B, out_features]
        """
        B, N, D = z.shape
        
        # Compute statistics for z
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            z_masked = z * mask_expanded
            count = mask.sum(dim=1, keepdim=True).clamp(min=1)
            
            z_mean = z_masked.sum(dim=1) / count
            z_var = ((z_masked - z_mean.unsqueeze(1)) ** 2 * mask_expanded).sum(dim=1) / count
            z_std = (z_var + 1e-6).sqrt()
        else:
            z_mean = z.mean(dim=1)
            z_std = z.std(dim=1) + 1e-6
        
        stats = [z_mean, z_std]
        
        # Add delta statistics
        if self.mode in ['mean_std_delta', 'full'] and delta_z is not None:
            delta_mask = mask[:, 1:] if mask is not None else None
            
            if delta_mask is not None:
                dm_expanded = delta_mask.unsqueeze(-1).float()
                delta_masked = delta_z * dm_expanded
                delta_count = delta_mask.sum(dim=1, keepdim=True).clamp(min=1)
                
                delta_mean = delta_masked.sum(dim=1) / delta_count
                delta_var = ((delta_masked - delta_mean.unsqueeze(1)) ** 2 * dm_expanded).sum(dim=1) / delta_count
                delta_std = (delta_var + 1e-6).sqrt()
            else:
                delta_mean = delta_z.mean(dim=1)
                delta_std = delta_z.std(dim=1) + 1e-6
            
            stats.extend([delta_mean, delta_std])
        
        # Add max/min
        if self.mode == 'full':
            if mask is not None:
                z_for_max = z.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                z_for_min = z.masked_fill(~mask.unsqueeze(-1), float('inf'))
                z_max = z_for_max.max(dim=1).values
                z_min = z_for_min.min(dim=1).values
            else:
                z_max = z.max(dim=1).values
                z_min = z.min(dim=1).values
            
            stats.extend([z_max, z_min])
        
        ecg_emb = torch.cat(stats, dim=-1)
        
        return ecg_emb

