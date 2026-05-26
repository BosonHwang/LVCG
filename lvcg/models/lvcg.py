"""
LVCG: Latent VCG Generator for ECG Representation Learning

This model learns ECG representations through VCG-based reconstruction with
dual supervision on structural and temporal dynamics.

Supports two variants:
- LVCG-GRU: StateGRU for state generation
- LVCG-TTT: LowRankStateGenerator with Direct State Prediction

Key Design Principles:
1. Single Shared Latent: Only one latent z_t sequence
2. Initialization + Causal Evolution:
   - z_0 = BeatEncoder(VCG_beat_1) (first complete beat)
   - z_{t+1}^{pred} = MLP(z_t) (direct state prediction)
3. Dual Supervision:
   - L_recon: Spatial/geometric consistency (z_t^{pred} -> ECG reconstruction)
   - L_temporal: State matching (MSE(z_pred, z_real)) - NOT delta!
4. Downstream Embedding:
   - emb_struct = z_0 (first complete beat state)
   - emb_dynamic = GRU.h_last OR flatten(U, V) (rhythm/dynamics)
   - ecg_emb = concat(emb_struct, emb_dynamic, emb_rhythm)

Architecture (Direct State Prediction):
    ECG -> VCG -> Beat Segment -> Strip incomplete beats
                                        |
                               V_beats_core [B, M, 3, P]
                               (beat_1 to beat_{N-2})
                                        |
                                  BeatEncoder
                                        |
                               states_core [B, M, D]
                                        |
         +------------------------------+------------------------------+
         |                              |                              |
    emb_struct                    TTT (Direct State)            states_real
    = states_core[:,0,:]               |                              |
    (beat_1)                      z_pred, states_pred            L_temporal
                                       |                         MSE(states_pred, states_real)
                                  BeatDecoder
                                       |
                                 V_hat_core [B, M, 3, P]
                                       |
                              Pad back beat_0, beat_{N-1}
                                       |
                                  Stitcher
                                       |
                                  V_hat [B, 3, T]
                                       |
                              GeomProj + ECGDecoder
                                       |
                                  E_hat [B, 12, T]
                                       |
                                  L_recon
    
Key Design (Avoids Shortcut Learning):
    - Predicts next STATE directly, NOT delta
    - L_temporal = MSE(z_pred, z_real), harder to cheat with constant output
    - BeatEncoder must output meaningful states for TTT to predict next one
"""

from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from ..data.angle import get_lead_directions
from ..data.beat_segmentation import BeatSegmenter
from .blocks.beat_modules import (
    BeatEncoder,
    BeatDecoder,
    BeatStitcher,
    GlobalRREmbedding,
)
from .blocks.ttt_state_gen import LowRankStateGenerator
from .vcg import VCGPseudoInverse, GeometricLeadProjection
from .blocks.decoder import ECGRefinementDecoder


# =============================================================================
# LVCG-specific modules
# =============================================================================

class StateGRU(nn.Module):
    """
    Causal state generator via GRU.
    
    This module generates state sequences purely based on state dynamics,
    without using RR interval embeddings.
    
    Input: state_0 [B, D]
    Output: 
        states_pred [B, N, D] - Generated state sequence
        h_last [B, H] - Last hidden state (for emb_dynamic)
    
    Generation Process:
        states_pred[0] = state_0
        for t in 1..N-1:
            h_t = GRU_step(states_pred[t-1])
            states_pred[t] = out_proj(h_t)
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU: state_dim -> hidden_dim
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,  # Causal
        )
        
        # Project GRU output back to state_dim (if hidden_dim != state_dim)
        if hidden_dim != state_dim:
            self.out_proj = nn.Linear(hidden_dim, state_dim)
        else:
            self.out_proj = nn.Identity()
    
    def forward(
        self,
        state_0: torch.Tensor,
        num_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate state sequence via GRU unrolling.
        
        Args:
            state_0: [B, D] - Initial state (BeatEncoder output)
            num_steps: N - Number of states to generate
        
        Returns:
            states_pred: [B, N, D] - Generated states (state_0, state_1, ..., state_{N-1})
            h_last: [B, H] - Last GRU hidden state (for emb_dynamic)
        """
        B = state_0.shape[0]
        device = state_0.device
        
        # Collect generated states
        states_pred_list = [state_0]  # state_0 is given
        
        # Initialize GRU hidden state
        h = None
        
        # Current input to GRU
        state_t = state_0
        
        for t in range(1, num_steps):
            # GRU step: state_t -> h_t
            state_t_input = state_t.unsqueeze(1)  # [B, 1, D]
            gru_out, h = self.gru(state_t_input, h)  # gru_out: [B, 1, H], h: [num_layers, B, H]
            
            # Project to state_dim
            state_t = self.out_proj(gru_out.squeeze(1))  # [B, D]
            
            states_pred_list.append(state_t)
        
        # Stack all states
        states_pred = torch.stack(states_pred_list, dim=1)  # [B, N, D]
        
        # Last hidden state (top layer)
        if h is not None:
            h_last = h[-1]  # [B, H]
        else:
            # If num_steps == 1, no GRU forward was done
            h_last = torch.zeros(B, self.hidden_dim, device=device)
        
        return states_pred, h_last


class LinearPredictor(nn.Module):
    """
    Simple linear state predictor (drop-in replacement for GRU).

    Predicts next state via a linear map:
        z_{t+1} = W z_t + b
    """

    def __init__(self, state_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.linear = nn.Linear(state_dim, state_dim)

    def forward(
        self,
        state_0: torch.Tensor,
        num_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate state sequence via linear unrolling.

        Returns:
            states_pred: [B, N, D]
            h_last: [B, D] (use last predicted state)
        """
        B = state_0.shape[0]
        device = state_0.device

        states_pred_list = [state_0]
        state_t = state_0

        for _ in range(1, num_steps):
            state_t = self.linear(state_t)
            states_pred_list.append(state_t)

        states_pred = torch.stack(states_pred_list, dim=1)  # [B, N, D]
        h_last = states_pred_list[-1] if num_steps > 0 else torch.zeros(B, self.state_dim, device=device)

        return states_pred, h_last


class ProjectionMLP(nn.Module):
    """
    Projection head for SSL (MoCo).
    
    Input: [B, in_dim]
    Output: [B, out_dim]
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# =============================================================================
# Main Model
# =============================================================================

@register_model("lvcg")
class LVCG(nn.Module):
    """
    LVCG: Latent VCG Generator for ECG Representation Learning
    
    Core Design:
    1. Single shared latent z_t
    2. z_0 = BeatEncoder(beat_0) + rr_emb_0
    3. z_t^{pred} = GRU(z_{t-1}) + rr_emb_t (causal generation)
    4. L_recon: z^{pred} -> ECG reconstruction
    5. L_temporal: Huber(delta_z^{pred}, sg(delta_z^{real}))
    6. emb_struct = z_0, emb_dynamic = GRU.h_last
    """
    
    def __init__(
        self,
        # Input config
        time_len: int = 1024,
        num_leads: int = 12,
        lead_order: str = 'mimic',
        
        # Beat config
        beat_len: int = 128,
        state_dim: int = 256,
        max_beats: int = 20,
        
        # BeatEncoder config (compact version)
        beat_encoder_stem: int = 48,
        beat_encoder_stages: List[int] = [96, 192, 256, 256],
        beat_encoder_blocks: List[int] = [3, 3, 3, 2],
        beat_encoder_kernel: int = 7,
        beat_encoder_dropout: float = 0.1,
        
        # StateGRU config (used when use_ttt=False)
        gru_hidden_dim: int = 256,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.1,
        temporal_type: str = "gru",
        
        # TTT config (used when use_ttt=True)
        use_ttt: bool = False,
        ttt_proj_dim: int = 32,
        ttt_rank: int = 4,
        ttt_base_lr: float = 0.01,
        
        # BeatDecoder config
        decoder_initial_channels: int = 128,
        decoder_hidden_channels: List[int] = [128, 64, 64, 32, 32],
        
        # RR embedding
        rr_num_basis: int = 8,
        
        # Projection for SSL
        proj_dim: int = 256,
        proj_hidden: int = 512,
        
        # ECG Refinement Decoder
        ecg_decoder_hidden: int = 128,
        ecg_decoder_layers: int = 2,
        
        # Other
        fs: int = 100,
        rr_lead_idx: int = 1,
    ):
        super().__init__()
        
        self.time_len = time_len
        self.num_leads = num_leads
        self.state_dim = state_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.rr_lead_idx = rr_lead_idx
        self.use_ttt = use_ttt
        self.temporal_type = temporal_type
        
        # ========== VCG Recovery ==========
        self.vcg_inverse = VCGPseudoInverse(eps=0.1)
        
        # Use direction vectors
        lead_directions = get_lead_directions(order=lead_order, as_tensor=True)
        self.lead_projection = GeometricLeadProjection(lead_directions=lead_directions)
        self.register_buffer('all_lead_directions', lead_directions)
        
        # ========== Beat Segmentation ==========
        self.beat_segmenter = BeatSegmenter(
            beat_len=beat_len,
            fs=fs,
            max_beats=max_beats,
        )
        
        # ========== Beat Encoder ==========
        self.beat_encoder = BeatEncoder(
            beat_len=beat_len,
            state_dim=state_dim,
            stem_channels=beat_encoder_stem,
            stage_channels=beat_encoder_stages,
            stage_blocks=beat_encoder_blocks,
            kernel_size=beat_encoder_kernel,
            dropout=beat_encoder_dropout,
        )
        
        # ========== Global RR Embedding ==========
        self.rhythm_dim = 128  # Fixed to 128 dimensions
        self.global_rr_embedding = GlobalRREmbedding(
            out_dim=self.rhythm_dim,
            max_beats=max_beats,
            num_basis=rr_num_basis,
        )
        
        # ========== State Generator (GRU or TTT) ==========
        if use_ttt:
            # LVCG-TTT: LowRankStateGenerator with Test-Time Training
            self.state_generator = LowRankStateGenerator(
                state_dim=state_dim,
                proj_dim=ttt_proj_dim,
                rank=ttt_rank,
                base_lr=ttt_base_lr,
            )
            # emb_dynamic dim for TTT = proj_dim * rank * 2 = 256
            self.emb_dynamic_dim = self.state_generator.emb_dim
        else:
            # LVCG-GRU: StateGRU or LinearPredictor for causal state generation
            if temporal_type == "linear":
                self.state_generator = LinearPredictor(state_dim=state_dim)
                self.emb_dynamic_dim = state_dim
            elif temporal_type == "gru":
                self.state_generator = StateGRU(
                    state_dim=state_dim,
                    hidden_dim=gru_hidden_dim,
                    num_layers=gru_num_layers,
                    dropout=gru_dropout,
                )
                # emb_dynamic dim for GRU = gru_hidden_dim
                self.emb_dynamic_dim = gru_hidden_dim
            else:
                raise ValueError(f"Unknown temporal_type: {temporal_type}. Use 'gru' or 'linear'.")
        
        # ========== Beat Decoder ==========
        self.beat_decoder = BeatDecoder(
            state_dim=state_dim,
            beat_len=beat_len,
            initial_channels=decoder_initial_channels,
            hidden_channels=decoder_hidden_channels,
        )
        
        # ========== Beat Stitcher ==========
        self.stitcher = BeatStitcher(
            beat_len=beat_len,
            target_len=time_len,
        )
        
        # ========== ECG Refinement Decoder ==========
        self.ecg_decoder = ECGRefinementDecoder(
            num_leads=num_leads,
            hidden_dim=ecg_decoder_hidden,
            num_layers=ecg_decoder_layers,
        )
        
        # ========== Projection MLPs (for SSL) ==========
        self.struct_proj = ProjectionMLP(
            in_dim=state_dim,
            out_dim=proj_dim,
            hidden_dim=proj_hidden,
        )
        self.dynamic_proj = ProjectionMLP(
            in_dim=self.emb_dynamic_dim,
            out_dim=proj_dim,
            hidden_dim=proj_hidden,
        )
        
        # ========== LayerNorm for Embeddings ==========
        # Standardize embeddings for downstream tasks
        self.norm_struct = nn.LayerNorm(state_dim)
        self.norm_dynamic = nn.LayerNorm(self.emb_dynamic_dim)
        self.norm_rhythm = nn.LayerNorm(self.rhythm_dim)
        
        # Output features = emb_struct + emb_dynamic + emb_rhythm
        self.out_features = state_dim + self.emb_dynamic_dim + self.rhythm_dim
    
    def _recover_vcg(
        self,
        ecg: torch.Tensor,
        visible_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Recover VCG from visible leads."""
        B, L, T = ecg.shape
        device = ecg.device
        num_visible = visible_indices.shape[1]
        
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_visible)
        visible_ecg = ecg[batch_idx, visible_indices]
        
        # Get direction vectors for visible leads
        visible_directions = self.all_lead_directions[visible_indices]  # [B, K, 3]
        
        vcg = self.vcg_inverse(visible_ecg, visible_directions)
        
        return vcg
    
    def forward_train(
        self,
        ecg: torch.Tensor,
        visible_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Supports two variants:
        - LVCG-GRU: StateGRU for state generation
        - LVCG-TTT: Autoregressive state prediction on core beats only
        
        Args:
            ecg: [B, 12, T] - Input ECG
            visible_indices: [B, K] - Indices of visible leads
        
        Returns:
            Dict containing:
            - recon: [B, 12, T] - Reconstructed ECG
            - states_pred: [B, M-2, D] - Predicted states (TTT) or [B, N-1, D] (GRU)
            - states_real: [B, M-2, D] - Real states (TTT) or [B, N-1, D] (GRU)
            - emb_struct, emb_dynamic, emb_rhythm: Three embeddings
            - ecg_emb: [B, D+emb_dynamic_dim+128] - Final embedding
        """
        B, L, T = ecg.shape
        device = ecg.device
        
        # 1. VCG Recovery
        vcg = self._recover_vcg(ecg, visible_indices)  # [B, 3, T]
        
        # 2. Beat Segmentation
        V_beats_full, rr_intervals_full, beat_mask_full = self.beat_segmenter(
            vcg, ecg, rr_lead_idx=self.rr_lead_idx
        )  # V_beats_full: [B, N, 3, P], rr_intervals_full: [B, N], beat_mask_full: [B, N]
        N = V_beats_full.shape[1]
        
        # Need at least 4 beats for TTT (2 context + 1 predict + boundary)
        assert N >= 4, f"Need at least 4 beats, got {N}"
        
        # ========== TTT: Strip incomplete beats ==========
        if self.use_ttt:
            # Strip beat_0 and beat_{N-1} (potentially incomplete)
            V_beats_core = V_beats_full[:, 1:-1, :, :].contiguous()  # [B, M, 3, P] where M = N-2
            rr_core = rr_intervals_full[:, 1:-1].contiguous()        # [B, M]
            beat_mask_core = beat_mask_full[:, 1:-1].contiguous()    # [B, M]
            M = V_beats_core.shape[1]
            
            # BeatEncoder only on core beats
            states_core = self.beat_encoder(V_beats_core)  # [B, M, D]
            
            # emb_struct: first complete beat (beat_1 in original, index 0 in core)
            state_base = states_core[:, 0, :]  # [B, D]
            V_base = V_beats_core[:, 0, :, :]  # [B, 3, P]
            
            # Decode state_base (for L_base loss)
            V_base_hat = self.beat_decoder(state_base.unsqueeze(1)).squeeze(1)  # [B, 3, P]
            
            # TTT Direct State Prediction
            ttt_out = self.state_generator(states_core)
            z_pred = ttt_out['z_pred']                          # [B, M, D]
            states_pred_for_loss = ttt_out['states_pred_for_loss']  # [B, M-2, D]
            states_real_for_loss = ttt_out['states_real_for_loss']  # [B, M-2, D]
            emb_dynamic = ttt_out['emb_dynamic']                # [B, 256]
            
            # Decode predicted states
            V_hat_core = self.beat_decoder(z_pred)  # [B, M, 3, P]
            
            # Pad back beat_0 and beat_{N-1} using boundary reconstructions
            V_hat_beat0 = V_hat_core[:, 0:1, :, :]   # [B, 1, 3, P]
            V_hat_beatN = V_hat_core[:, -1:, :, :]   # [B, 1, 3, P]
            V_hat_beats = torch.cat([V_hat_beat0, V_hat_core, V_hat_beatN], dim=1)  # [B, N, 3, P]
            
            # For temporal loss
            beat_mask_for_loss = beat_mask_core[:, 2:]  # [B, M-2]
            
            # For beat_level_loss: use core beats
            V_beats = V_beats_core
            states_real = states_core
            
        else:
            # LVCG-GRU: StateGRU without rr_emb
            V_beats = V_beats_full
            beat_mask = beat_mask_full
            states_real = self.beat_encoder(V_beats)  # [B, N, D]
            
            # state_base: first complete beat (beat_1)
            state_base = states_real[:, 1, :]  # [B, D]
            V_base = V_beats[:, 1, :, :]  # [B, 3, P]
            
            # Decode state_base (for L_base loss)
            V_base_hat = self.beat_decoder(state_base.unsqueeze(1)).squeeze(1)  # [B, 3, P]
            
            num_gen_steps = N - 1  # Generate N-1 states
            states_pred, h_last = self.state_generator(state_base, num_steps=num_gen_steps)
            # states_pred: [B, N-1, D], h_last: [B, H]
            emb_dynamic = h_last  # [B, H] - GRU last hidden state
            
            # Decode predicted states
            V_hat_beats_pred = self.beat_decoder(states_pred)  # [B, N-1, 3, P]
            
            # beat_0 placeholder: decode states_real[:, 0, :]
            state_beat0 = states_real[:, 0:1, :]  # [B, 1, D]
            V_hat_beat0 = self.beat_decoder(state_beat0)  # [B, 1, 3, P]
            
            # Concatenate: [beat_0, beat_1, ..., beat_{N-1}]
            V_hat_beats = torch.cat([V_hat_beat0, V_hat_beats_pred], dim=1)  # [B, N, 3, P]
            
            # For temporal loss (state matching for GRU)
            states_real_for_loss = states_real[:, 1:, :]  # [B, N-1, D]
            states_pred_for_loss = states_pred  # [B, N-1, D]
            beat_mask_for_loss = beat_mask[:, 1:]  # [B, N-1]
        
        # ========== Reconstruction Branch ==========
        V_hat = self.stitcher(V_hat_beats, rr_intervals_full, beat_mask_full)  # [B, 3, T]
        
        # Geometric projection to 12 leads
        E_hat_geom = self.lead_projection(V_hat)  # [B, 12, T]
        
        # ECG Refinement Decoder
        E_hat = self.ecg_decoder(E_hat_geom)  # [B, 12, T]
        
        # ========== Embeddings ==========
        emb_struct = state_base  # [B, D] - First complete beat's state
        
        # Global RR embedding
        emb_rhythm = self.global_rr_embedding(rr_intervals_full, beat_mask_full)  # [B, 128]
        
        # LayerNorm: standardize all embeddings
        emb_struct = self.norm_struct(emb_struct)    # [B, D]
        emb_dynamic = self.norm_dynamic(emb_dynamic)  # [B, emb_dynamic_dim]
        emb_rhythm = self.norm_rhythm(emb_rhythm)    # [B, 128]
        
        # Projections for SSL
        emb_struct_proj = self.struct_proj(emb_struct)    # [B, proj_dim]
        emb_dynamic_proj = self.dynamic_proj(emb_dynamic)  # [B, proj_dim]
        
        # Final embedding = struct + dynamic + rhythm
        ecg_emb = torch.cat([emb_struct, emb_dynamic, emb_rhythm], dim=-1)
        
        return {
            # Reconstruction
            'recon': E_hat,  # [B, 12, T]
            'recon_geom': E_hat_geom,  # [B, 12, T]
            'vcg_hat': V_hat,  # [B, 3, T]
            'V_beats': V_beats,
            'V_hat_beats': V_hat_beats,  # [B, N, 3, P]
            
            # state_base related (for L_base loss)
            'state_base': state_base,  # [B, D]
            'V_base': V_base,  # [B, 3, P]
            'V_base_hat': V_base_hat,  # [B, 3, P]
            
            # States for temporal loss
            'states_real': states_real_for_loss,
            'states_pred': states_pred_for_loss,
            
            # Embeddings
            'emb_struct': emb_struct,  # [B, D]
            'emb_dynamic': emb_dynamic,
            'emb_rhythm': emb_rhythm,  # [B, 128]
            'emb_struct_proj': emb_struct_proj,  # [B, proj_dim]
            'emb_dynamic_proj': emb_dynamic_proj,  # [B, proj_dim]
            'ecg_emb': ecg_emb,
            
            # Auxiliary
            'rr_intervals': rr_intervals_full,
            'beat_mask': beat_mask_for_loss,
            'beat_mask_full': beat_mask_full,
            
            # For beat_level_loss
            'states_core': states_real,
        }
    
    def forward_inference(
        self,
        ecg: torch.Tensor,
        use_all_leads: bool = True,
    ) -> torch.Tensor:
        """
        Inference: Extract ecg_emb = concat(emb_struct, emb_dynamic, emb_rhythm)
        
        Args:
            ecg: [B, 12, T] - Input ECG
            use_all_leads: Whether to use all leads for VCG recovery
        
        Returns:
            ecg_emb: [B, D+emb_dynamic_dim+128] - Final embedding
        """
        B, L, T = ecg.shape
        device = ecg.device
        
        # VCG recovery
        if use_all_leads:
            all_directions = self.all_lead_directions.unsqueeze(0).expand(B, -1, -1)
            vcg = self.vcg_inverse(ecg, all_directions)
        else:
            visible_indices = torch.tensor([[0, 1, 6]], device=device).expand(B, -1)
            vcg = self._recover_vcg(ecg, visible_indices)
        
        # Beat segmentation
        V_beats_full, rr_intervals, beat_mask = self.beat_segmenter(
            vcg, ecg, rr_lead_idx=self.rr_lead_idx
        )
        N = V_beats_full.shape[1]
        
        if self.use_ttt:
            # LVCG-TTT: Strip incomplete beats
            if N < 4:
                # Fallback: not enough beats
                state_base = torch.zeros(B, self.state_dim, device=device)
                emb_dynamic = torch.zeros(B, self.emb_dynamic_dim, device=device)
            else:
                V_beats_core = V_beats_full[:, 1:-1, :, :].contiguous()  # [B, M, 3, P]
                states_core = self.beat_encoder(V_beats_core)  # [B, M, D]
                
                state_base = states_core[:, 0, :]  # [B, D]
                
                ttt_out = self.state_generator(states_core)
                emb_dynamic = ttt_out['emb_dynamic']  # [B, 256]
        else:
            # LVCG-GRU: Encode all beats
            states_real = self.beat_encoder(V_beats_full)  # [B, N, D]
            
            if N < 2:
                state_base = states_real[:, 0, :]  # [B, D]
                emb_dynamic = torch.zeros(B, self.gru_hidden_dim, device=device)
            else:
                state_base = states_real[:, 1, :]  # [B, D]
                num_gen_steps = N - 1
                _, emb_dynamic = self.state_generator(state_base, num_steps=num_gen_steps)
        
        # Embeddings
        emb_struct = state_base  # [B, D]
        emb_rhythm = self.global_rr_embedding(rr_intervals, beat_mask)  # [B, 128]
        
        # LayerNorm
        emb_struct = self.norm_struct(emb_struct)
        emb_dynamic = self.norm_dynamic(emb_dynamic)
        emb_rhythm = self.norm_rhythm(emb_rhythm)
        
        # Final embedding
        ecg_emb = torch.cat([emb_struct, emb_dynamic, emb_rhythm], dim=-1)
        
        return ecg_emb
    
    @torch.no_grad()
    def ext_ecg_emb(
        self,
        ecg: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Compatible interface for downstream tasks."""
        ecg_emb = self.forward_inference(ecg, use_all_leads=True)
        
        if normalize:
            ecg_emb = F.normalize(ecg_emb, p=2, dim=-1)
        
        return ecg_emb
    
    def forward(
        self,
        ecg: torch.Tensor,
        visible_indices: Optional[torch.Tensor] = None,
    ):
        """Default forward."""
        if visible_indices is not None:
            return self.forward_train(ecg, visible_indices)
        else:
            return self.forward_inference(ecg)
    
    @classmethod
    def from_config(cls, cfg):
        """Instantiate from config."""
        model_cfg = cfg.model
        return cls(
            time_len=int(cfg.data.get("time_len", 1024)),
            num_leads=int(model_cfg.get("num_leads", 12)),
            lead_order=str(model_cfg.get("lead_order", "mimic")),
            beat_len=int(model_cfg.get("beat_len", 128)),
            state_dim=int(model_cfg.get("state_dim", 256)),
            max_beats=int(model_cfg.get("max_beats", 20)),
            beat_encoder_stem=int(model_cfg.get("beat_encoder_stem", 48)),
            beat_encoder_stages=list(model_cfg.get("beat_encoder_stages", [96, 192, 256, 256])),
            beat_encoder_blocks=list(model_cfg.get("beat_encoder_blocks", [3, 3, 3, 2])),
            beat_encoder_kernel=int(model_cfg.get("beat_encoder_kernel", 7)),
            beat_encoder_dropout=float(model_cfg.get("beat_encoder_dropout", 0.1)),
            gru_hidden_dim=int(model_cfg.get("gru_hidden_dim", 256)),
            gru_num_layers=int(model_cfg.get("gru_num_layers", 2)),
            gru_dropout=float(model_cfg.get("gru_dropout", 0.1)),
            temporal_type=str(model_cfg.get("temporal_type", "gru")),
            use_ttt=bool(model_cfg.get("use_ttt", False)),
            ttt_proj_dim=int(model_cfg.get("ttt_proj_dim", 32)),
            ttt_rank=int(model_cfg.get("ttt_rank", 4)),
            ttt_base_lr=float(model_cfg.get("ttt_base_lr", 0.01)),
            decoder_initial_channels=int(model_cfg.get("decoder_initial_channels", 128)),
            decoder_hidden_channels=list(model_cfg.get("decoder_hidden_channels", [128, 64, 64, 32, 32])),
            rr_num_basis=int(model_cfg.get("rr_num_basis", 8)),
            proj_dim=int(model_cfg.get("proj_dim", 256)),
            proj_hidden=int(model_cfg.get("proj_hidden", 512)),
            ecg_decoder_hidden=int(model_cfg.get("ecg_decoder_hidden", 128)),
            ecg_decoder_layers=int(model_cfg.get("ecg_decoder_layers", 2)),
            fs=int(cfg.data.get("fs", 100)),
            rr_lead_idx=int(model_cfg.get("rr_lead_idx", 1)),
        )


# =============================================================================
# Loss Functions
# =============================================================================

def masked_reconstruction_loss(E_hat: torch.Tensor, ecg: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Masked leads MSE loss for reconstruction.
    
    Args:
        E_hat: [B, 12, T] - Reconstructed ECG
        ecg: [B, 12, T] - Original ECG
        mask: [B, 12] - True = masked (needs reconstruction)
    
    Returns:
        loss: Scalar MSE loss on masked leads only
    """
    mask_expanded = mask.unsqueeze(-1).float()  # [B, 12, 1]
    diff = (E_hat - ecg) ** 2  # [B, 12, T]
    
    # Sum over masked positions, average
    loss = (diff * mask_expanded).sum() / mask_expanded.sum().clamp(min=1) / ecg.shape[-1]
    return loss


def temporal_loss(z_pred: torch.Tensor, z_real: torch.Tensor, beat_mask: torch.Tensor) -> torch.Tensor:
    """
    Temporal dynamics loss for GRU: compare predicted vs real state differences.
    
    Args:
        z_pred: [B, N, D] - GRU-generated state sequence
        z_real: [B, N, D] - BeatEncoder-encoded real states
        beat_mask: [B, N] - Valid beat mask
    
    Returns:
        loss: Huber(delta_z_pred, sg(delta_z_real))
    """
    # Compute differences
    delta_z_pred = z_pred[:, 1:, :] - z_pred[:, :-1, :]  # [B, N-1, D]
    delta_z_real = z_real[:, 1:, :] - z_real[:, :-1, :]  # [B, N-1, D]
    
    # Stop gradient on target
    delta_z_real = delta_z_real.detach()
    
    # Mask for valid transitions
    trans_mask = beat_mask[:, 1:] & beat_mask[:, :-1]  # [B, N-1]
    
    # Huber loss (robust to outliers)
    loss = F.huber_loss(delta_z_pred, delta_z_real, reduction='none')  # [B, N-1, D]
    
    # Apply mask
    mask_expanded = trans_mask.unsqueeze(-1).float()  # [B, N-1, 1]
    loss = (loss * mask_expanded).sum() / mask_expanded.sum().clamp(min=1)
    
    return loss


def delta_loss(
    delta_pred: torch.Tensor,
    delta_real: torch.Tensor,
    beat_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Delta prediction loss for TTT.
    
    Args:
        delta_pred: [B, M-2, D] - Predicted deltas from TTT
        delta_real: [B, M-2, D] - Real deltas (already detached)
        beat_mask: [B, M-2] - Valid beat mask (optional)
    
    Returns:
        loss: MSE(delta_pred, delta_real)
    """
    if beat_mask is not None:
        mask_expanded = beat_mask.unsqueeze(-1).float()  # [B, M-2, 1]
        loss = ((delta_pred - delta_real) ** 2 * mask_expanded).sum()
        loss = loss / mask_expanded.sum().clamp(min=1)
    else:
        loss = F.mse_loss(delta_pred, delta_real)
    
    return loss


def beat_level_loss(V_hat_beats: torch.Tensor, V_beats: torch.Tensor, beat_mask: torch.Tensor) -> torch.Tensor:
    """
    Optional beat-level VCG reconstruction loss.
    
    Args:
        V_hat_beats: [B, N, 3, P] - Decoded VCG beats
        V_beats: [B, N, 3, P] - Original VCG beats
        beat_mask: [B, N] - Valid beat mask
    
    Returns:
        loss: MSE on valid beats
    """
    mask_expanded = beat_mask.unsqueeze(-1).unsqueeze(-1).float()  # [B, N, 1, 1]
    diff = (V_hat_beats - V_beats) ** 2  # [B, N, 3, P]
    loss = (diff * mask_expanded).sum() / mask_expanded.sum().clamp(min=1)
    return loss


def base_beat_loss(V_base_hat: torch.Tensor, V_base: torch.Tensor) -> torch.Tensor:
    """
    Base beat (z_base) reconstruction loss.
    
    Ensures high encoding quality for the first complete beat (beat_1),
    which serves as a reliable anchor for GRU generation.
    
    Args:
        V_base_hat: [B, 3, P] - z_base decoded VCG beat
        V_base: [B, 3, P] - beat_1 original VCG
    
    Returns:
        loss: MSE loss
    """
    loss = F.mse_loss(V_base_hat, V_base)
    return loss

