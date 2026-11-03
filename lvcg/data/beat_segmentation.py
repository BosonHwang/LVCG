"""
Beat Segmentation for LVCG.

Pipeline Role: VCG [B, 3, T] + ECG [B, 12, T] -> V_beats [B, N, 3, P], rr_intervals [B, N], beat_mask [B, N]

Uses R-peak detection on Lead II (or specified lead) to segment beats.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import find_peaks


class BeatSegmenter(nn.Module):
    """
    Segment VCG into individual beats using R-peak detection.
    
    Input: VCG [B, 3, T], ECG [B, 12, T]
    Output: V_beats [B, N, 3, P], rr_intervals [B, N], beat_mask [B, N]
    
    Key Design:
    - R-peak detection on Lead II (or specified lead)
    - Resample each beat to fixed length P
    - Store original RR intervals for stitching
    """
    
    def __init__(
        self,
        beat_len: int = 128,
        fs: int = 100,
        max_beats: int = 20,
        min_rr_samples: int = 30,  # Minimum RR interval in samples
        max_rr_samples: int = 300,  # Maximum RR interval in samples
    ):
        """
        Args:
            beat_len: Output beat length P (after resampling)
            fs: Sampling frequency
            max_beats: Maximum number of beats to extract
            min_rr_samples: Minimum RR interval in samples
            max_rr_samples: Maximum RR interval in samples
        """
        super().__init__()
        self.beat_len = beat_len
        self.fs = fs
        self.max_beats = max_beats
        self.min_rr_samples = min_rr_samples
        self.max_rr_samples = max_rr_samples
    
    def _detect_r_peaks(self, lead_signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks in a single lead signal.
        
        Args:
            lead_signal: [T] - Single lead ECG signal
        
        Returns:
            r_peaks: Array of R-peak indices
        """
        # Normalize signal
        signal = lead_signal - np.mean(lead_signal)
        std = np.std(signal)
        if std > 1e-6:
            signal = signal / std
        
        # Find peaks with minimum distance and height
        min_distance = self.min_rr_samples
        peaks, properties = find_peaks(
            signal,
            distance=min_distance,
            height=0.3,  # Require positive peaks above threshold
            prominence=0.5,
        )
        
        # If too few peaks, try with lower threshold
        if len(peaks) < 3:
            peaks, _ = find_peaks(
                signal,
                distance=min_distance,
                height=0.1,
            )
        
        return peaks
    
    def _get_beat_boundaries(self, r_peaks: np.ndarray, T: int) -> List[Tuple[int, int]]:
        """
        Get beat boundaries from R-peaks, ensuring full coverage of [0, T].
        
        Coverage strategy:
        - First segment: [0, r_peaks[0]) - always include to cover start
        - Middle segments: [R_i, R_{i+1}) - filter by RR bounds
        - Last segment: [r_peaks[-1], T) - always include to cover end
        
        Args:
            r_peaks: Array of R-peak indices
            T: Total signal length
        
        Returns:
            List of (start, end) tuples for each beat
        """
        if len(r_peaks) < 2:
            # Fallback: divide signal into equal parts
            num_beats = min(self.max_beats, T // self.min_rr_samples)
            if num_beats < 1:
                num_beats = 1
            beat_len = T // num_beats
            return [(i * beat_len, (i + 1) * beat_len) for i in range(num_beats)]
        
        # Build all segments
        all_segments = []
        
        # First segment: [0, r_peaks[0]) - always include if > 0
        first_peak = int(r_peaks[0])
        if first_peak > 0:
            all_segments.append((0, first_peak, True))  # True = boundary segment
        
        # Middle segments: [R_i, R_{i+1})
        for i in range(len(r_peaks) - 1):
            start = int(r_peaks[i])
            end = int(r_peaks[i + 1])
            rr_len = end - start
            if self.min_rr_samples <= rr_len <= self.max_rr_samples:
                all_segments.append((start, end, False))  # False = middle segment
        
        # Last segment: [r_peaks[-1], T) - always include if > 0
        last_peak = int(r_peaks[-1])
        if T - last_peak > 0:
            all_segments.append((last_peak, T, True))  # True = boundary segment
        
        # Priority: boundary segments first, then middle segments (up to max_beats)
        boundary_segs = [(s, e) for s, e, is_boundary in all_segments if is_boundary]
        middle_segs = [(s, e) for s, e, is_boundary in all_segments if not is_boundary]
        
        # Reserve slots for boundaries
        remaining_slots = self.max_beats - len(boundary_segs)
        selected_middle = middle_segs[:max(0, remaining_slots)]
        
        # Combine and sort by start position
        boundaries = boundary_segs + selected_middle
        boundaries.sort(key=lambda x: x[0])
        
        return boundaries
    
    def forward(
        self,
        vcg: torch.Tensor,
        ecg: torch.Tensor,
        rr_lead_idx: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Segment VCG into beats.
        
        Args:
            vcg: [B, 3, T] - VCG signal
            ecg: [B, 12, T] - Original ECG (for R-peak detection)
            rr_lead_idx: Lead index for R-peak detection (default: 1 = Lead II)
        
        Returns:
            V_beats: [B, N, 3, P] - Segmented and resampled VCG beats
            rr_intervals: [B, N] - Original RR intervals in samples
            beat_mask: [B, N] - Valid beat mask (True = valid)
        
        Shape Flow:
            VCG [B, 3, T] -> detect R-peaks on ECG[rr_lead_idx]
                          -> segment into beats
                          -> resample each beat to P
                          -> V_beats [B, N, 3, P]
        """
        B, C, T = vcg.shape
        device = vcg.device
        dtype = vcg.dtype
        N = self.max_beats
        P = self.beat_len
        
        # Get R-peaks for each sample
        ecg_np = ecg.detach().cpu().numpy()
        all_boundaries = []
        
        for b in range(B):
            lead_signal = ecg_np[b, rr_lead_idx]
            r_peaks = self._detect_r_peaks(lead_signal)
            boundaries = self._get_beat_boundaries(r_peaks, T)
            all_boundaries.append(boundaries)
        
        # Allocate outputs
        V_beats = torch.zeros((B, N, 3, P), dtype=dtype, device=device)
        rr_intervals = torch.zeros((B, N), dtype=torch.float32, device=device)
        beat_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
        
        # Segment and resample
        for b in range(B):
            boundaries = all_boundaries[b][:N]
            for n, (start, end) in enumerate(boundaries):
                # Extract beat from VCG
                beat_vcg = vcg[b, :, start:end]  # [3, rr_len]
                rr_len = end - start
                
                # Resample to beat_len
                beat_vcg = beat_vcg.unsqueeze(0)  # [1, 3, rr_len]
                beat_resampled = F.interpolate(
                    beat_vcg, size=P, mode='linear', align_corners=False
                ).squeeze(0)  # [3, P]
                
                V_beats[b, n] = beat_resampled
                rr_intervals[b, n] = float(rr_len)
                beat_mask[b, n] = True
        
        return V_beats, rr_intervals, beat_mask

