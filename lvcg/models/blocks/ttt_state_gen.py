"""
Low-Rank State Generator with Test-Time Training (TTT).

Architecture: Direct State Prediction (NOT delta!)
- Predicts next state directly: z_{t+1} = MLP(z_t)
- L_temporal = ||z_pred - z_real||^2
- Avoids shortcut learning where BeatEncoder outputs constants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class LowRankStateGenerator(nn.Module):
    """
    TTT-based state generator with low-rank MLP.
    
    Architecture (Direct State Prediction):
        - Input: z_t [B, D]
        - Output: z_{t+1} [B, D] (next state directly)
        - z_{t+1} = MLP(z_t)  (NOT z_t + delta!)
    
    MLP Structure:
        z [D] -> proj_in [D'] -> LowRank(U,V) -> GELU -> proj_out [D] -> z_next
        
    TTT Update:
        - Computes gradient w.r.t. state prediction error
        - Updates U, V online at each step
        
    Embedding:
        emb_dynamic = flatten(U_final, V_final) = D' * rank * 2 = 256 dim
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        proj_dim: int = 32,
        rank: int = 4,
        base_lr: float = 0.1,
    ):
        """
        Args:
            state_dim: State dimension D (default: 256)
            proj_dim: Projection dimension D' (default: 32)
            rank: Low-rank dimension (default: 4)
            base_lr: Fixed learning rate for TTT updates (default: 0.1)
        """
        super().__init__()
        self.state_dim = state_dim
        self.proj_dim = proj_dim
        self.rank = rank
        self.base_lr = base_lr
        
        # Projection layers (not updated by TTT)
        self.proj_in = nn.Linear(state_dim, proj_dim, bias=False)
        self.proj_out = nn.Linear(proj_dim, state_dim, bias=False)
        
        # Low-rank parameters (updated by TTT)
        # U: [D', rank], V: [D', rank]
        # Initialize with scale=0.5 for stable output magnitude
        self.U_init = nn.Parameter(torch.randn(proj_dim, rank) * 0.5)
        self.V_init = nn.Parameter(torch.randn(proj_dim, rank) * 0.5)
        
        # emb_dynamic dimension: D' * rank * 2 = 32 * 4 * 2 = 256
        self.emb_dim = proj_dim * rank * 2
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize projection weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def _lowrank_forward(
        self, 
        z_proj: torch.Tensor,  # [B, D']
        U: torch.Tensor,       # [B, D', rank]
        V: torch.Tensor,       # [B, D', rank]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Low-rank MLP forward pass.
        
        Args:
            z_proj: [B, D'] - Projected input
            U: [B, D', rank] - Low-rank factor U
            V: [B, D', rank] - Low-rank factor V
        
        Returns:
            h: [B, D'] - Output after GELU
            pre_act: [B, D'] - Pre-activation (for gradient computation)
        """
        # W = U @ V.T: [B, D', D']
        W = torch.einsum('bdr,ber->bde', U, V)
        # pre_act = z_proj @ W: [B, D']
        pre_act = torch.einsum('bd,bde->be', z_proj, W)
        # h = GELU(pre_act): [B, D']
        h = F.gelu(pre_act)
        return h, pre_act

    def _predict_next_state(
        self,
        z_t: torch.Tensor,  # [B, D]
        U: torch.Tensor,    # [B, D', rank]
        V: torch.Tensor,    # [B, D', rank]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state directly from current state.
        
        Args:
            z_t: [B, D] - Current state
            U, V: [B, D', rank] - Current low-rank factors
        
        Returns:
            z_next: [B, D] - Predicted next state (NOT delta!)
            z_proj: [B, D'] - Projected input (for gradient)
            pre_act: [B, D'] - Pre-activation (for gradient)
        """
        z_proj = self.proj_in(z_t)  # [B, D']
        h, pre_act = self._lowrank_forward(z_proj, U, V)  # [B, D']
        z_next = self.proj_out(h)  # [B, D] - Direct next state prediction
        return z_next, z_proj, pre_act

    def _compute_grad(
        self,
        z_pred: torch.Tensor,    # [B, D]
        z_target: torch.Tensor,  # [B, D]
        z_proj: torch.Tensor,    # [B, D']
        pre_act: torch.Tensor,   # [B, D']
        U: torch.Tensor,         # [B, D', rank]
        V: torch.Tensor,         # [B, D', rank]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients for U and V using chain rule.
        
        Loss: L = ||z_pred - z_target||^2
        
        Args:
            z_pred: [B, D] - Predicted next state
            z_target: [B, D] - Target next state (real)
            z_proj: [B, D'] - Projected input
            pre_act: [B, D'] - Pre-activation
            U, V: [B, D', rank] - Current factors
        
        Returns:
            grad_U: [B, D', rank] - Gradient for U
            grad_V: [B, D', rank] - Gradient for V
        """
        # dL/d(z_pred) = 2 * (z_pred - z_target): [B, D]
        grad_output = 2.0 * (z_pred - z_target)
        
        # GELU derivative: d/dx GELU(x) = CDF(x) + x * PDF(x)
        SQRT_2 = 1.4142135623730951
        SQRT_2PI = 2.5066282746310002
        cdf = 0.5 * (1.0 + torch.erf(pre_act / SQRT_2))
        pdf = torch.exp(-0.5 * pre_act * pre_act) / SQRT_2PI
        gelu_grad = cdf + pre_act * pdf  # [B, D']
        
        # dL/dh = dL/d(z_pred) @ proj_out.weight * gelu_grad: [B, D']
        grad_h = (grad_output @ self.proj_out.weight) * gelu_grad
        
        # dL/dW = z_proj.T @ grad_h: [B, D', D']
        grad_W = torch.einsum('bd,be->bde', z_proj, grad_h)
        
        # dL/dU = dL/dW @ V: [B, D', rank]
        grad_U = torch.einsum('bde,ber->bdr', grad_W, V)
        
        # dL/dV = dL/dW.T @ U: [B, D', rank]
        grad_V = torch.einsum('bde,bdr->ber', grad_W, U)
        
        return grad_U, grad_V

    def _update_UV(
        self,
        grad_U: torch.Tensor,  # [B, D', rank]
        grad_V: torch.Tensor,  # [B, D', rank]
        U: torch.Tensor,       # [B, D', rank]
        V: torch.Tensor,       # [B, D', rank]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update U, V with gradient descent (with gradient clipping).
        
        Args:
            grad_U, grad_V: Gradients
            U, V: Current factors
        
        Returns:
            U_new, V_new: Updated factors
        """
        # Gradient clipping (normalize if norm > 1)
        grad_norm_U = grad_U.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        grad_norm_V = grad_V.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        grad_U = grad_U / grad_norm_U.clamp(min=1.0)
        grad_V = grad_V / grad_norm_V.clamp(min=1.0)
        
        # Fixed learning rate update
        U_new = U - self.base_lr * grad_U
        V_new = V - self.base_lr * grad_V
        
        return U_new, V_new

    def forward(
        self, 
        states_core: torch.Tensor,  # [B, M, D]
    ) -> Dict[str, torch.Tensor]:
        """
        Direct state prediction with TTT (NOT delta prediction!).
        
        Data flow:
            - Context: z_0, z_1 are real states (beat_1, beat_2)
            - Autoregressive: z_2, z_3, ... are predicted directly
            - z_{t+1} = MLP(z_t)  (NO delta accumulation!)
        
        Args:
            states_core: [B, M, D] - States of complete beats only
                         M = N-2 where N is total beats
                         Excludes incomplete beat_0 and beat_{N-1}
        
        Returns:
            Dict containing:
                z_pred: [B, M, D] - Predicted state sequence
                        [z_0, z_1, z_2_pred, z_3_pred, ...]
                         real  real  <-- predicted -->
                states_pred_for_loss: [B, M-2, D] - Predicted states for L_temporal
                states_real_for_loss: [B, M-2, D] - Real states for L_temporal
                emb_dynamic: [B, 256] - flatten(U_final, V_final)
        """
        B, M, D = states_core.shape
        device = states_core.device
        
        # Initialize U, V per sample
        U = self.U_init.unsqueeze(0).expand(B, -1, -1).clone()  # [B, D', rank]
        V = self.V_init.unsqueeze(0).expand(B, -1, -1).clone()  # [B, D', rank]
        
        # Context: z_0, z_1 are real (need at least 2 beats for context)
        assert M >= 3, f"Need at least 3 complete beats, got {M}"
        z_0 = states_core[:, 0, :]  # [B, D] - beat_1
        z_1 = states_core[:, 1, :]  # [B, D] - beat_2
        
        # Initialize prediction list with context
        z_pred_list = [z_0, z_1]  # First two are real
        z_pred_for_loss_list = []  # Predicted states for L_temporal
        
        # Autoregressive prediction from z_1 onwards
        z_current = z_1  # Start from beat_2
        
        for t in range(1, M - 1):  # t=1 to M-2, predict z_2 to z_{M-1}
            # 1. Predict next state DIRECTLY (no delta!)
            z_next_pred, z_proj, pre_act = self._predict_next_state(z_current, U, V)
            # z_next_pred: [B, D] - predicted z_{t+1}
            z_pred_for_loss_list.append(z_next_pred)
            
            # 2. Get target: TRUE next state
            z_target = states_core[:, t + 1, :]  # [B, D]
            
            # 3. Update U, V with TTT (gradient on state prediction error)
            grad_U, grad_V = self._compute_grad(
                z_next_pred, z_target.detach(), z_proj, pre_act, U, V
            )
            U, V = self._update_UV(grad_U, grad_V, U, V)
            
            # 4. Use prediction as next input (autoregressive)
            z_pred_list.append(z_next_pred)
            z_current = z_next_pred
        
        # Stack results
        z_pred = torch.stack(z_pred_list, dim=1)  # [B, M, D]
        states_pred_for_loss = torch.stack(z_pred_for_loss_list, dim=1)  # [B, M-2, D]
        states_real_for_loss = states_core[:, 2:, :]  # [B, M-2, D]
        
        # emb_dynamic = flatten(U_final, V_final)
        emb_dynamic = torch.cat([
            U.reshape(B, -1),  # [B, D' * rank]
            V.reshape(B, -1),  # [B, D' * rank]
        ], dim=-1)  # [B, 256]
        
        return {
            'z_pred': z_pred,                       # [B, M, D]
            'states_pred_for_loss': states_pred_for_loss,  # [B, M-2, D]
            'states_real_for_loss': states_real_for_loss,  # [B, M-2, D]
            'emb_dynamic': emb_dynamic,             # [B, 256]
        }

    def forward_inference(
        self,
        states_core: torch.Tensor,  # [B, M, D]
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode (same as forward, TTT still active).
        
        Args:
            states_core: [B, M, D] - States of complete beats
        
        Returns:
            Same as forward()
        """
        return self.forward(states_core)

