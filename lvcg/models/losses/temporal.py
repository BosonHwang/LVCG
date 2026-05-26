"""
Normalized Huber Loss for temporal difference prediction.

Key Design:
- Target is stop_grad (detached)
- Per-dimension normalization for stable gradients
- Huber loss for robustness to outliers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NormalizedHuberLoss(nn.Module):
    """
    Huber loss with per-dimension normalization.
    
    Normalizes both prediction and target before computing loss.
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        eps: float = 1e-6,
        normalize: bool = True,
    ):
        """
        Args:
            delta: Huber loss threshold
            eps: Small constant for numerical stability
            normalize: Whether to apply per-dim normalization
        """
        super().__init__()
        self.delta = delta
        self.eps = eps
        self.normalize = normalize
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute normalized Huber loss.
        
        Args:
            pred: [B, N, D] - Predictions
            target: [B, N, D] - Targets (will be detached internally)
            mask: [B, N] - Valid positions mask (optional)
        
        Returns:
            loss: Scalar loss value
        """
        # Ensure target is detached
        target = target.detach()
        
        if self.normalize:
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                count = mask.sum() * pred.shape[-1]
                
                pred_masked = pred * mask_expanded
                target_masked = target * mask_expanded
                
                pred_mean = pred_masked.sum() / count.clamp(min=1)
                target_mean = target_masked.sum() / count.clamp(min=1)
                
                pred_std = ((pred_masked - pred_mean) ** 2).sum() / count.clamp(min=1)
                pred_std = (pred_std + self.eps).sqrt()
                target_std = ((target_masked - target_mean) ** 2).sum() / count.clamp(min=1)
                target_std = (target_std + self.eps).sqrt()
            else:
                pred_mean = pred.mean()
                pred_std = pred.std() + self.eps
                target_mean = target.mean()
                target_std = target.std() + self.eps
            
            pred_norm = (pred - pred_mean) / pred_std
            target_norm = (target - target_mean) / target_std
        else:
            pred_norm = pred
            target_norm = target
        
        # Huber loss
        loss = F.huber_loss(pred_norm, target_norm, reduction='none', delta=self.delta)
        
        # Apply mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            loss = loss * mask_expanded
            loss = loss.sum() / mask_expanded.sum().clamp(min=1)
        else:
            loss = loss.mean()
        
        return loss

