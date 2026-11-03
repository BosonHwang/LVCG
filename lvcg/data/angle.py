"""
Lead direction vectors and reorder utilities for ECG multi-lead reconstruction.

This module provides:
- Standard 12-lead ECG direction vectors based on clinical ECG geometry
- Dataset-specific lead reordering functions (MIMIC <-> PTBXL)

Coordinate System (Frank/Standard ECG):
- X-axis: right -> left (positive = left)
- Y-axis: superior -> inferior (positive = inferior/feet)
- Z-axis: anterior -> posterior (positive = posterior)

Lead Geometry:
- Limb leads (I, II, III, aVR, aVL, aVF): Frontal plane (XY plane, Z=0)
- Precordial leads (V1-V6): Transverse plane (XZ plane with slight Y tilt)

References:
- Einthoven's triangle for limb leads
- Standard precordial electrode positions
"""

from typing import Dict, List, Union
import numpy as np
import torch


# =============================================================================
# Lead Direction Vectors (Standard ECG Geometry)
# =============================================================================

# Lead names for each order
LEAD_NAMES_PTBXL = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'III', 'aVR', 'aVL', 'aVF']
LEAD_NAMES_MIMIC = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Standard Lead Direction Vectors in MIMIC order
# Based on clinical ECG geometry:
# - Limb leads: Frontal plane angles (Einthoven triangle)
#   - Lead I:   0 deg (horizontal, pointing left)
#   - Lead II:  60 deg (left-inferior)
#   - Lead III: 120 deg (right-inferior)
#   - aVR: -150 deg (right-superior)
#   - aVL: -30 deg (left-superior)
#   - aVF: 90 deg (inferior)
# - Precordial leads: Transverse plane with ~10 deg inferior tilt
#   - V1: -110 deg (right-anterior)
#   - V2: -70 deg (anterior-right)
#   - V3: -40 deg (anterior)
#   - V4: -10 deg (anterior-left)
#   - V5: 20 deg (left-anterior)
#   - V6: 50 deg (lateral)
LEAD_DIRECTIONS_MIMIC = np.array([
    # Limb leads (Frontal plane, Z=0)
    [ 1.00000,  0.00000,  0.00000],  # I   - 0 deg, horizontal left
    [ 0.50000,  0.86603,  0.00000],  # II  - 60 deg, left-inferior
    [-0.50000,  0.86603,  0.00000],  # III - 120 deg, right-inferior
    [-0.86603, -0.50000,  0.00000],  # aVR - -150 deg, right-superior
    [ 0.86603, -0.50000,  0.00000],  # aVL - -30 deg, left-superior
    [ 0.00000,  1.00000,  0.00000],  # aVF - 90 deg, inferior
    # Precordial leads (Transverse plane with ~10 deg inferior tilt)
    [-0.33682,  0.17365,  0.92542],  # V1  - right-anterior
    [ 0.33682,  0.17365,  0.92542],  # V2  - anterior-right
    [ 0.75441,  0.17365,  0.63302],  # V3  - anterior
    [ 0.96985,  0.17365,  0.17101],  # V4  - anterior-left
    [ 0.92542,  0.17365, -0.33682],  # V5  - left-anterior
    [ 0.63302,  0.17365, -0.75441],  # V6  - lateral
], dtype=np.float32)

# =============================================================================
# Lead Order Mappings
# =============================================================================

# MIMIC standard order: [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
# PTBXL order:          [I, II, V1, V2, V3, V4, V5, V6, III, aVR, aVL, aVF]

# Index mapping: MIMIC[i] -> PTBXL[MIMIC_TO_PTBXL_ORDER[i]]
# To convert MIMIC-ordered data to PTBXL order: new_data = data[MIMIC_TO_PTBXL_ORDER]
MIMIC_TO_PTBXL_ORDER = [0, 1, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5]

# Inverse: PTBXL[i] -> MIMIC[PTBXL_TO_MIMIC_ORDER[i]]
PTBXL_TO_MIMIC_ORDER = [0, 1, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7]

# Mapping dictionary for flexible reordering
ORDER_MAPPINGS: Dict[str, Dict[str, List[int]]] = {
    'mimic': {
        'ptbxl': MIMIC_TO_PTBXL_ORDER,
    },
    'ptbxl': {
        'mimic': PTBXL_TO_MIMIC_ORDER,
    }
}

# Pre-compute PTBXL order directions
LEAD_DIRECTIONS_PTBXL = LEAD_DIRECTIONS_MIMIC[MIMIC_TO_PTBXL_ORDER]


# =============================================================================
# Direction Vector Functions
# =============================================================================

def get_lead_directions(order: str = 'mimic', as_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    Get lead direction vectors in specified order.
    
    These are the standard 12-lead ECG direction vectors based on clinical geometry.
    
    Args:
        order: Lead order ('ptbxl' or 'mimic')
        as_tensor: If True, return torch.Tensor; else np.ndarray
    
    Returns:
        Lead direction vectors, shape [12, 3]
    """
    if order.lower() == 'mimic':
        directions = LEAD_DIRECTIONS_MIMIC.copy()
    elif order.lower() == 'ptbxl':
        directions = LEAD_DIRECTIONS_PTBXL.copy()
    else:
        raise ValueError(f"Unknown order: {order}. Choose from: 'mimic', 'ptbxl'")
    
    if as_tensor:
        return torch.from_numpy(directions)
    return directions


def compute_lead_directions(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Compute 3D direction vectors from lead angles (for dynamic/batched angles).
    
    NOTE: For standard 12-lead ECG, prefer get_lead_directions() which uses
    pre-computed correct direction vectors.
    
    This function is kept for cases where angles are dynamically computed
    or vary per sample.
    
    Coordinate convention:
        - theta: angle in frontal plane (from X-axis, counterclockwise)
        - phi: elevation angle (tilt from frontal plane)
    
    Args:
        theta: Frontal plane angles [L] or [B, L] (radians)
        phi: Elevation angles [L] or [B, L] (radians, 0 = frontal plane)
    
    Returns:
        Direction vectors, shape [L, 3] or [B, L, 3]
        u = [cos(phi)*cos(theta), cos(phi)*sin(theta), sin(phi)]
    """
    u_x = torch.cos(phi) * torch.cos(theta)
    u_y = torch.cos(phi) * torch.sin(theta)
    u_z = torch.sin(phi)
    return torch.stack([u_x, u_y, u_z], dim=-1)


def compute_lead_directions_np(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    NumPy version of compute_lead_directions.
    
    Args:
        theta: Frontal plane angles [L] or [B, L] (radians)
        phi: Elevation angles [L] or [B, L] (radians)
    
    Returns:
        Direction vectors, shape [L, 3] or [B, L, 3]
    """
    u_x = np.cos(phi) * np.cos(theta)
    u_y = np.cos(phi) * np.sin(theta)
    u_z = np.sin(phi)
    return np.stack([u_x, u_y, u_z], axis=-1)


# =============================================================================
# Lead Reordering Functions
# =============================================================================

def reorder_leads(
    ecg: Union[np.ndarray, torch.Tensor],
    source: str,
    target: str
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reorder ECG leads from source dataset order to target order.
    
    Pipeline Role: Standardize lead ordering across different datasets
                   before model processing.
    
    Input Semantics:
        - ecg: ECG signal with leads in source dataset order
        - source: Source dataset name ('mimic', 'ptbxl')
        - target: Target order name ('mimic', 'ptbxl')
    
    Output Semantics:
        - ECG signal with leads reordered to target order
    
    Args:
        ecg: ECG signal, shape [B, 12, T] or [12, T]
        source: Source dataset name (e.g., 'mimic', 'ptbxl')
        target: Target order name (e.g., 'ptbxl', 'mimic')
    
    Returns:
        Reordered ECG with same shape as input
    
    Raises:
        ValueError: If source or target is not recognized
    """
    if source == target:
        return ecg
    
    source_lower = source.lower()
    target_lower = target.lower()
    
    if source_lower not in ORDER_MAPPINGS:
        raise ValueError(f"Unknown source order: {source}. Available: {list(ORDER_MAPPINGS.keys())}")
    if target_lower not in ORDER_MAPPINGS[source_lower]:
        raise ValueError(f"No mapping from {source} to {target}")
    
    order = ORDER_MAPPINGS[source_lower][target_lower]
    
    # Handle both numpy and torch
    if isinstance(ecg, np.ndarray):
        if ecg.ndim == 2:
            # [12, T] -> [12, T]
            return ecg[order]
        elif ecg.ndim == 3:
            # [B, 12, T] -> [B, 12, T]
            return ecg[:, order, :]
        else:
            raise ValueError(f"ECG must be 2D or 3D, got {ecg.ndim}D")
    else:
        # torch.Tensor
        if ecg.ndim == 2:
            return ecg[order]
        elif ecg.ndim == 3:
            return ecg[:, order, :]
        else:
            raise ValueError(f"ECG must be 2D or 3D, got {ecg.ndim}D")


def reorder_directions(
    directions: Union[np.ndarray, torch.Tensor],
    source: str,
    target: str
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reorder lead direction vectors from source order to target order.
    
    Args:
        directions: Lead directions, shape [12, 3] or [B, 12, 3]
        source: Source order name
        target: Target order name
    
    Returns:
        Reordered directions with same shape as input
    """
    if source == target:
        return directions
    
    source_lower = source.lower()
    target_lower = target.lower()
    
    if source_lower not in ORDER_MAPPINGS:
        raise ValueError(f"Unknown source order: {source}")
    if target_lower not in ORDER_MAPPINGS[source_lower]:
        raise ValueError(f"No mapping from {source} to {target}")
    
    order = ORDER_MAPPINGS[source_lower][target_lower]
    
    if isinstance(directions, np.ndarray):
        if directions.ndim == 2:
            return directions[order]
        elif directions.ndim == 3:
            return directions[:, order, :]
        else:
            raise ValueError(f"Directions must be 2D or 3D, got {directions.ndim}D")
    else:
        if directions.ndim == 2:
            return directions[order]
        elif directions.ndim == 3:
            return directions[:, order, :]
        else:
            raise ValueError(f"Directions must be 2D or 3D, got {directions.ndim}D")


# =============================================================================
# Backward Compatibility: Angle-based functions
# =============================================================================

def directions_to_angles(directions: np.ndarray) -> np.ndarray:
    """
    Convert direction vectors to (theta, phi) angles.
    
    theta: angle in XY plane from X-axis (frontal plane angle)
    phi: elevation angle from XY plane
    
    Args:
        directions: [L, 3] or [B, L, 3] direction vectors
    
    Returns:
        angles: [L, 2] or [B, L, 2] with [theta, phi] per lead
    """
    u_x = directions[..., 0]
    u_y = directions[..., 1]
    u_z = directions[..., 2]
    
    # theta = atan2(y, x) - angle in XY plane
    theta = np.arctan2(u_y, u_x)
    
    # phi = asin(z) - elevation angle
    # Clamp to avoid numerical issues
    u_z_clamped = np.clip(u_z, -1.0, 1.0)
    phi = np.arcsin(u_z_clamped)
    
    return np.stack([theta, phi], axis=-1)


# Pre-compute angles from directions for backward compatibility
LEAD_ANGLES_MIMIC = directions_to_angles(LEAD_DIRECTIONS_MIMIC)
LEAD_ANGLES_PTBXL = directions_to_angles(LEAD_DIRECTIONS_PTBXL)

# Backward compatibility alias (deprecated, use LEAD_ANGLES_PTBXL instead)
LEAD_ANGLES_PTBXL_ORDER = LEAD_ANGLES_PTBXL


def get_lead_angles(order: str = 'mimic', as_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    Get lead angles in specified order.
    
    NOTE: These angles are derived from the standard direction vectors.
    For VCG operations, prefer using get_lead_directions() directly.
    
    Args:
        order: Lead order ('ptbxl' or 'mimic')
        as_tensor: If True, return torch.Tensor; else np.ndarray
    
    Returns:
        Lead angles, shape [12, 2] with [theta, phi] per lead
        - theta: frontal plane angle (radians)
        - phi: elevation angle (radians)
    """
    if order.lower() == 'mimic':
        angles = LEAD_ANGLES_MIMIC.copy()
    elif order.lower() == 'ptbxl':
        angles = LEAD_ANGLES_PTBXL.copy()
    else:
        raise ValueError(f"Unknown order: {order}")
    
    if as_tensor:
        return torch.from_numpy(angles.astype(np.float32))
    return angles.astype(np.float32)


def reorder_angles(
    angles: Union[np.ndarray, torch.Tensor],
    source: str,
    target: str
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reorder lead angles from source order to target order.
    
    Args:
        angles: Lead angles, shape [12, 2] or [B, 12, 2]
        source: Source order name
        target: Target order name
    
    Returns:
        Reordered angles with same shape as input
    """
    if source == target:
        return angles
    
    source_lower = source.lower()
    target_lower = target.lower()
    
    if source_lower not in ORDER_MAPPINGS:
        raise ValueError(f"Unknown source order: {source}")
    if target_lower not in ORDER_MAPPINGS[source_lower]:
        raise ValueError(f"No mapping from {source} to {target}")
    
    order = ORDER_MAPPINGS[source_lower][target_lower]
    
    if isinstance(angles, np.ndarray):
        if angles.ndim == 2:
            return angles[order]
        elif angles.ndim == 3:
            return angles[:, order, :]
        else:
            raise ValueError(f"Angles must be 2D or 3D, got {angles.ndim}D")
    else:
        if angles.ndim == 2:
            return angles[order]
        elif angles.ndim == 3:
            return angles[:, order, :]
        else:
            raise ValueError(f"Angles must be 2D or 3D, got {angles.ndim}D")


def get_visible_lead_angles(
    lead_angles: torch.Tensor,
    visible_indices: torch.Tensor
) -> torch.Tensor:
    """
    Extract angles for visible leads given indices.
    
    Args:
        lead_angles: All lead angles [12, 2]
        visible_indices: Indices of visible leads [B, K]
    
    Returns:
        Visible lead angles [B, K, 2]
    """
    # lead_angles: [12, 2]
    # visible_indices: [B, K]
    # Output: [B, K, 2]
    return lead_angles[visible_indices]  # Advanced indexing


def get_visible_lead_directions(
    lead_directions: torch.Tensor,
    visible_indices: torch.Tensor
) -> torch.Tensor:
    """
    Extract direction vectors for visible leads given indices.
    
    Args:
        lead_directions: All lead directions [12, 3]
        visible_indices: Indices of visible leads [B, K]
    
    Returns:
        Visible lead directions [B, K, 3]
    """
    # lead_directions: [12, 3]
    # visible_indices: [B, K]
    # Output: [B, K, 3]
    return lead_directions[visible_indices]  # Advanced indexing

