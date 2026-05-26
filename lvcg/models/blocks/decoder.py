"""
ECG Refinement Decoder for post-projection signal enhancement.

This module provides:
- ECGRefinementDecoder: Lightweight residual network to refine geometric projections

Key Design Principles:
- Residual learning: Learns corrections on top of geometric projection
- Lightweight: Small capacity to preserve VCG bottleneck semantics
- Lead-independent: Each lead processed independently with shared parameters
"""

from typing import List

import torch
import torch.nn as nn


class ECGRefinementDecoder(nn.Module):
    """
    ECG refinement decoder for post-projection enhancement.
    
    Pipeline Role: Apply learnable residual corrections to geometric
                   VCG->ECG projections.
    
    Input Semantics:
        - x: Geometric projection output [B, 12, T]
          - Raw ECG estimate from VCG projection
    
    Output Semantics:
        - out: Refined ECG [B, 12, T]
          - Geometry + learned residual correction
    
    Key Design:
        - Residual architecture: out = x + residual(x)
        - Lightweight capacity: Prevents decoder from dominating reconstruction
        - Lead-independent: All leads share parameters
    """
    
    def __init__(
        self,
        num_leads: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 2,
        kernel_size: int = 3
    ):
        """
        Args:
            num_leads: Number of ECG leads (not used directly, kept for API)
            hidden_dim: Hidden channels in conv layers
            num_layers: Number of conv layers
            kernel_size: Convolution kernel size
        """
        super().__init__()
        self.num_leads = num_leads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build conv layers
        layers: List[nn.Module] = []
        in_dim = 1  # Each lead is processed as 1-channel signal
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            padding = (kernel_size - 1) // 2  # Same padding
            
            layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding))
            
            # Add activation for all but last layer
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
            
            in_dim = out_dim
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual refinement to geometric projection.
        
        Args:
            x: Geometric projection [B, 12, T]
        
        Returns:
            out: Refined ECG [B, 12, T]
        
        Shape Flow:
            x: [B, L, T]
            -> reshape: [B*L, 1, T]
            -> net: [B*L, 1, T]
            -> residual add
            -> reshape: [B, L, T]
        """
        B, L, T = x.shape
        
        # Reshape for lead-independent processing
        # [B, L, T] -> [B*L, 1, T]
        # Use contiguous() to ensure memory layout is correct for view
        x_flat = x.contiguous().view(B * L, 1, T)
        
        # Compute residual
        residual = self.net(x_flat)  # [B*L, 1, T]
        
        # Add residual
        out = x_flat + residual  # [B*L, 1, T]
        
        # Reshape back
        out = out.view(B, L, T)  # [B, L, T]
        
        return out


class DepthwiseConvDecoder(nn.Module):
    """
    Depthwise separable convolution decoder for efficiency.
    
    Uses depthwise separable convolutions for larger receptive field
    with fewer parameters.
    """
    
    def __init__(
        self,
        num_leads: int = 12,
        hidden_dim: int = 64,
        kernel_size: int = 7,
        expansion: int = 2
    ):
        """
        Args:
            num_leads: Number of leads
            hidden_dim: Hidden dimension
            kernel_size: Depthwise conv kernel size
            expansion: Channel expansion factor
        """
        super().__init__()
        self.num_leads = num_leads
        
        padding = (kernel_size - 1) // 2
        
        # Depthwise conv (per-channel)
        self.depthwise = nn.Conv1d(
            num_leads, num_leads,
            kernel_size=kernel_size,
            padding=padding,
            groups=num_leads  # Depthwise: each channel separate
        )
        
        # Pointwise expansion
        self.pointwise_up = nn.Conv1d(num_leads, num_leads * expansion, kernel_size=1)
        
        # Activation
        self.act = nn.GELU()
        
        # Pointwise reduction
        self.pointwise_down = nn.Conv1d(num_leads * expansion, num_leads, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, T]
        
        Returns:
            out: [B, L, T]
        """
        # x: [B, L, T]
        residual = x
        
        # Depthwise conv
        y = self.depthwise(x)  # [B, L, T]
        
        # Pointwise expansion
        y = self.pointwise_up(y)  # [B, L*expansion, T]
        y = self.act(y)
        
        # Pointwise reduction
        y = self.pointwise_down(y)  # [B, L, T]
        
        # Residual connection
        out = residual + y
        
        return out


class MultiScaleDecoder(nn.Module):
    """
    Multi-scale convolution decoder with different kernel sizes.
    
    Captures both local (small kernel) and contextual (large kernel)
    information for refinement.
    """
    
    def __init__(
        self,
        num_leads: int = 12,
        hidden_dim: int = 32,
        kernel_sizes: List[int] = [3, 7, 15]
    ):
        """
        Args:
            num_leads: Number of leads
            hidden_dim: Hidden dimension per branch
            kernel_sizes: List of kernel sizes for multi-scale branches
        """
        super().__init__()
        self.num_leads = num_leads
        self.num_scales = len(kernel_sizes)
        
        # Multi-scale branches
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            padding = (ks - 1) // 2
            branch = nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size=ks, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_dim, 1, kernel_size=1)
            )
            self.branches.append(branch)
        
        # Fusion weights
        self.fusion = nn.Conv1d(self.num_scales, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, T]
        
        Returns:
            out: [B, L, T]
        """
        B, L, T = x.shape
        
        # Process each lead independently
        x_flat = x.view(B * L, 1, T)  # [B*L, 1, T]
        
        # Multi-scale features
        scale_outputs = []
        for branch in self.branches:
            scale_out = branch(x_flat)  # [B*L, 1, T]
            scale_outputs.append(scale_out)
        
        # Stack scales: [B*L, num_scales, T]
        multi_scale = torch.cat(scale_outputs, dim=1)
        
        # Fuse scales: [B*L, 1, T]
        residual = self.fusion(multi_scale)
        
        # Add residual
        out = x_flat + residual
        
        # Reshape back
        out = out.view(B, L, T)
        
        return out

