"""
Loss utilities for LVCG training.
"""

import torch
from typing import Tuple


def random_lead_mask(
    batch_size: int,
    num_leads: int = 12,
    num_visible: int = 3,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random lead masks for training.
    
    Args:
        batch_size: Number of samples
        num_leads: Total number of leads (default: 12)
        num_visible: Number of visible leads (default: 3)
        device: Target device
    
    Returns:
        visible_indices: [B, K] - Indices of visible leads
        mask: [B, L] - Boolean mask (True = masked/hidden)
    """
    if device is None:
        device = torch.device('cpu')
    
    visible_indices = []
    for _ in range(batch_size):
        perm = torch.randperm(num_leads, device=device)
        visible_indices.append(perm[:num_visible])
    
    visible_indices = torch.stack(visible_indices, dim=0)  # [B, K]
    
    # Create mask: True = masked (not visible)
    mask = torch.ones((batch_size, num_leads), dtype=torch.bool, device=device)
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_visible)
    mask[batch_idx, visible_indices] = False
    
    return visible_indices, mask


def masked_reconstruction_loss(
    E_hat: torch.Tensor,
    ecg: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute masked reconstruction loss (MSE on masked leads only).
    
    Args:
        E_hat: [B, 12, T] - Reconstructed ECG
        ecg: [B, 12, T] - Original ECG
        mask: [B, 12] - Boolean mask (True = masked, needs reconstruction)
    
    Returns:
        loss: Scalar MSE loss on masked leads
    """
    # Expand mask to match time dimension
    mask_expanded = mask.unsqueeze(-1).float()  # [B, 12, 1]
    
    # Compute squared error
    diff = (E_hat - ecg) ** 2  # [B, 12, T]
    
    # Apply mask and compute mean
    masked_diff = diff * mask_expanded  # [B, 12, T]
    num_masked = mask_expanded.sum() * ecg.shape[-1]
    
    loss = masked_diff.sum() / num_masked.clamp(min=1)
    
    return loss

