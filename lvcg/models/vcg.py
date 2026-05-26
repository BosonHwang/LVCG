"""
VCG (Vectorcardiogram) modules for ECG multi-lead reconstruction.

This module provides:
- VCGPseudoInverse: Recover VCG from visible ECG leads via geometric pseudo-inverse
- GeometricLeadProjection: Project VCG to 12-lead ECG via geometric projection

Key Design Principles:
- No learnable parameters in VCG recovery/projection (pure geometry)
- Numerical stability via regularization in pseudo-inverse
- Lead-agnostic: works with any subset of visible leads
- Prefer direction vectors over angles for numerical stability
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..data.angle import (
    compute_lead_directions,
    get_lead_directions,
    LEAD_DIRECTIONS_MIMIC,
    LEAD_DIRECTIONS_PTBXL,
)


class VCGPseudoInverse(nn.Module):
    """
    VCG recovery via pseudo-inverse from visible ECG leads.
    
    Pipeline Role: Convert K visible ECG leads to 3D VCG trajectory
                   using geometric pseudo-inverse (no learnable parameters).
    
    Input Semantics:
        - S: K visible ECG lead signals [B, K, T]
        - U: Direction vectors for K leads [B, K, 3]
        (or theta/phi angles for backward compatibility)
    
    Output Semantics:
        - VCG: 3D heart vector trajectory [B, 3, T]
    
    Mathematical Formulation:
        Each ECG lead is a linear projection: s_i(t) = u_i^T * v(t)
        Given K leads S and direction matrix U (K x 3):
            VCG = U^+ @ S
        where U^+ = (U^T U + eps*I)^{-1} U^T is the regularized pseudo-inverse.
    
    Key Design:
        - No learnable parameters (pure geometric computation)
        - Regularization (eps) for numerical stability
        - Works with any K >= 3 visible leads
    """
    
    def __init__(self, eps: float = 1e-6):
        """
        Args:
            eps: Regularization coefficient for pseudo-inverse stability
        """
        super().__init__()
        self.eps = eps
    
    def forward_from_directions(
        self,
        S: torch.Tensor,
        U: torch.Tensor
    ) -> torch.Tensor:
        """
        Recover VCG from visible leads using direction vectors (preferred API).
        
        Args:
            S: Visible lead signals [B, K, T] where K is number of visible leads
            U: Direction vectors [B, K, 3] for visible leads
        
        Returns:
            VCG: 3D trajectory [B, 3, T]
        
        Shape Flow:
            S: [B, K, T]
            U: [B, K, 3] (direction vectors)
            U^+: [B, 3, K] (pseudo-inverse)
            VCG = U^+ @ S: [B, 3, T]
        """
        B, K, T = S.shape
        device = S.device
        dtype = S.dtype
        
        assert U.shape == (B, K, 3), f"U shape mismatch: expected {(B, K, 3)}, got {U.shape}"
        
        # Compute pseudo-inverse U_pinv: [B, 3, K]
        # U_pinv = (U^T U + eps*I)^{-1} U^T
        Ut = U.transpose(1, 2)           # [B, 3, K]
        UtU = Ut @ U                     # [B, 3, 3]
        
        # Add regularization for numerical stability
        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # [1, 3, 3]
        UtU_reg = UtU + self.eps * eye   # [B, 3, 3]
        
        # Compute inverse
        UtU_inv = torch.linalg.inv(UtU_reg)  # [B, 3, 3]
        U_pinv = UtU_inv @ Ut            # [B, 3, K]
        
        # Recover VCG: [B, 3, K] @ [B, K, T] -> [B, 3, T]
        VCG = U_pinv @ S
        
        return VCG
    
    def forward(
        self,
        S: torch.Tensor,
        theta_or_U: torch.Tensor,
        phi: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Recover VCG from visible leads via pseudo-inverse.
        
        Supports two calling conventions:
        1. forward(S, U) - with direction vectors [B, K, 3] (preferred)
        2. forward(S, theta, phi) - with angles [B, K] each (backward compatible)
        
        Args:
            S: Visible lead signals [B, K, T] where K is number of visible leads
            theta_or_U: Either direction vectors [B, K, 3] or theta angles [B, K]
            phi: Azimuth angles [B, K] (only if theta_or_U is theta)
        
        Returns:
            VCG: 3D trajectory [B, 3, T]
        """
        B, K, T = S.shape
        
        # Detect calling convention
        if theta_or_U.ndim == 3 and theta_or_U.shape[-1] == 3:
            # Direction vectors provided directly
            U = theta_or_U
        elif phi is not None:
            # Angles provided (backward compatibility)
            theta = theta_or_U
            assert theta.shape == (B, K), f"theta shape mismatch: expected {(B, K)}, got {theta.shape}"
            assert phi.shape == (B, K), f"phi shape mismatch: expected {(B, K)}, got {phi.shape}"
            U = compute_lead_directions(theta, phi)  # [B, K, 3]
        else:
            raise ValueError(
                "Either provide direction vectors U [B, K, 3] or both theta and phi [B, K]"
            )
        
        return self.forward_from_directions(S, U)


class GeometricLeadProjection(nn.Module):
    """
    Project VCG to multi-lead ECG via geometric projection.
    
    Pipeline Role: Convert 3D VCG trajectory to L-lead ECG signals
                   using fixed lead direction vectors (no learnable parameters).
    
    Input Semantics:
        - VCG: 3D heart vector trajectory [B, 3, T]
    
    Output Semantics:
        - ECG: L-lead ECG signals [B, L, T]
    
    Mathematical Formulation:
        For each lead l: ecg_l(t) = u_l^T @ vcg(t)
        where u_l is the unit direction vector for lead l.
    
    Key Design:
        - No learnable parameters (pure geometric projection)
        - Lead directions are pre-computed and stored as buffer
        - Supports any number of leads with provided directions
    """
    
    def __init__(
        self, 
        lead_directions: Optional[torch.Tensor] = None,
        lead_angles: Optional[torch.Tensor] = None,
        order: str = 'mimic'
    ):
        """
        Initialize with direction vectors (preferred) or angles (backward compatible).
        
        Args:
            lead_directions: [L, 3] tensor of direction vectors (preferred)
            lead_angles: [L, 2] tensor with [theta, phi] for each lead (backward compatible)
            order: Lead order ('mimic' or 'ptbxl'), used if neither directions nor angles provided
        """
        super().__init__()
        
        if lead_directions is not None:
            # Preferred: direction vectors provided directly
            assert lead_directions.ndim == 2 and lead_directions.shape[1] == 3, \
                f"lead_directions must be [L, 3], got {lead_directions.shape}"
            U = lead_directions
        elif lead_angles is not None:
            # Backward compatible: angles provided
            assert lead_angles.ndim == 2 and lead_angles.shape[1] == 2, \
                f"lead_angles must be [L, 2], got {lead_angles.shape}"
            theta = lead_angles[:, 0]  # [L]
            phi = lead_angles[:, 1]    # [L]
            U = compute_lead_directions(theta, phi)
        else:
            # Default: use standard directions based on order
            U = get_lead_directions(order=order, as_tensor=True)
        
        # Register as buffer (not learnable, but moves with model)
        self.register_buffer('U', U)  # [L, 3]
        self.num_leads = U.shape[0]
    
    def forward(self, VCG: torch.Tensor) -> torch.Tensor:
        """
        Project VCG to multi-lead ECG.
        
        Args:
            VCG: 3D trajectory [B, 3, T]
        
        Returns:
            ECG: L-lead signals [B, L, T]
        
        Shape Flow:
            VCG: [B, 3, T]
            U: [L, 3]
            ECG = einsum('lc,bct->blt'): [B, L, T]
        """
        assert VCG.ndim == 3 and VCG.shape[1] == 3, \
            f"VCG must be [B, 3, T], got {VCG.shape}"
        
        # U: [L, 3], VCG: [B, 3, T]
        # ECG_l(t) = sum_c U[l, c] * VCG[:, c, t]
        ECG = torch.einsum('lc,bct->blt', self.U, VCG)  # [B, L, T]
        
        return ECG


# =============================================================================
# Legacy VCG Generator (kept for backward compatibility)
# =============================================================================

class VCGGenerator(nn.Module):
    """
    Generate latent VCG trajectories V [B, 3, T'] from state W [B, D].
    
    DEPRECATED: This is the old TTT-based VCG generator.
    For the new reconstruction task, use VCGPseudoInverse + GeometricLeadProjection.

    1) Role: render a shared latent source (not ECG) from W.
    2) I/O: W [B, D] -> V [B, 3, T'] via low-DOF basis A(W) @ B.
    3) Pipeline: feeds LeadProjection; regularizers constrain geometry/dynamics.
    """

    def __init__(self, state_dim: int, basis_k: int = 32, time_len: int = 256):
        super().__init__()
        self.state_dim = int(state_dim)
        self.basis_k = int(basis_k)
        self.time_len = int(time_len)
        hidden = max(64, state_dim * 2)
        self.A_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3 * basis_k),
        )
        # Shared temporal basis across all samples
        self.B = nn.Parameter(torch.randn(basis_k, time_len) * 0.1)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        assert W.ndim == 2, "W must be [B, D]"
        assert W.shape[1] == self.state_dim, "state_dim mismatch"
        B = W.shape[0]
        A = self.A_net(W).reshape(B, 3, self.basis_k)  # [B, 3, K]
        V = torch.matmul(A, self.B)  # [B, 3, T']
        return V

    @staticmethod
    def regularizer_smoothness(V: torch.Tensor) -> torch.Tensor:
        """Penalize curvature via second differences along time."""
        d1 = V[..., 1:] - V[..., :-1]
        d2 = d1[..., 1:] - d1[..., :-1]
        return (d2 ** 2).mean()

    @staticmethod
    def regularizer_energy(V: torch.Tensor) -> torch.Tensor:
        """Penalize overall energy to prevent scale drift."""
        return (V ** 2).mean()

    @staticmethod
    def regularizer_loop_closure(V: torch.Tensor) -> torch.Tensor:
        """Encourage start ~ end if applicable (optional)."""
        start = V[..., 0]
        end = V[..., -1]
        return ((start - end) ** 2).mean()

