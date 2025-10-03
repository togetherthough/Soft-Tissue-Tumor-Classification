"""Saliency-based crop proposal generation and NMS for Stage-2"""

import numpy as np
import torch
from scipy import ndimage
from typing import List, Tuple


def peaks_from_saliency(
    saliency: np.ndarray,
    top_p: float = 0.005,
    min_distance: int = 8
) -> List[Tuple[int, int, int]]:
    """Extract peak coordinates from saliency map.
    
    Args:
        saliency: 3D saliency map (D, H, W) in [0, 1]
        top_p: Top percentile threshold for peak detection
        min_distance: Minimum distance for local maxima detection
        
    Returns:
        List of (z, y, x) coordinates sorted by saliency value (descending)
    """
    # Threshold by top percentile
    threshold = np.quantile(saliency, 1.0 - top_p)
    mask = saliency >= threshold
    
    # Find local maxima
    coords = local_maxima_3d(saliency, mask, min_distance)
    
    # Sort by saliency value
    values = [saliency[c] for c in coords]
    sorted_indices = np.argsort(values)[::-1]
    coords = [coords[i] for i in sorted_indices]
    
    return coords


def local_maxima_3d(
    volume: np.ndarray,
    mask: np.ndarray,
    min_distance: int = 8
) -> List[Tuple[int, int, int]]:
    """Find local maxima in 3D volume.
    
    Args:
        volume: 3D array
        mask: Binary mask where to look for maxima
        min_distance: Minimum distance between maxima
        
    Returns:
        List of (z, y, x) coordinates
    """
    # Apply maximum filter
    max_filtered = ndimage.maximum_filter(volume, size=min_distance)
    
    # Maxima are where original equals max-filtered
    maxima = (volume == max_filtered) & mask
    
    # Get coordinates
    coords = np.argwhere(maxima)
    coords = [(int(c[0]), int(c[1]), int(c[2])) for c in coords]
    
    return coords


def nms_3d(
    coords: List[Tuple[int, int, int]],
    scores: List[float],
    distance: int = 16
) -> List[Tuple[int, int, int]]:
    """Non-maximum suppression in 3D.
    
    Greedy NMS based on Euclidean distance in voxel units.
    
    Args:
        coords: List of (z, y, x) coordinates
        scores: Corresponding scores (higher is better)
        distance: Minimum distance threshold
        
    Returns:
        Filtered list of coordinates
    """
    if len(coords) == 0:
        return []
    
    # Sort by score descending
    sorted_indices = np.argsort(scores)[::-1]
    coords = [coords[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    keep = []
    coords_array = np.array(coords)
    
    for i, coord in enumerate(coords):
        # Check distance to all kept coords
        if len(keep) == 0:
            keep.append(i)
            continue
        
        kept_coords = coords_array[keep]
        distances = np.sqrt(np.sum((kept_coords - coord) ** 2, axis=1))
        
        if np.all(distances >= distance):
            keep.append(i)
    
    return [coords[i] for i in keep]


def random_diverse_centers(
    volume_shape: Tuple[int, int, int],
    n_centers: int,
    margin: int = 48
) -> List[Tuple[int, int, int]]:
    """Generate random diverse centers for crops.
    
    Used when saliency doesn't provide enough peaks.
    
    Args:
        volume_shape: Shape of volume (D, H, W)
        n_centers: Number of centers to generate
        margin: Margin from edges
        
    Returns:
        List of (z, y, x) coordinates
    """
    centers = []
    
    for _ in range(n_centers):
        z = np.random.randint(margin, volume_shape[0] - margin)
        y = np.random.randint(margin, volume_shape[1] - margin)
        x = np.random.randint(margin, volume_shape[2] - margin)
        centers.append((z, y, x))
    
    return centers


def get_crop_centers_from_saliency(
    saliency: torch.Tensor,
    K: int = 8,
    nms_distance: int = 16,
    top_p: float = 0.005,
    volume_shape: Tuple[int, int, int] = (192, 192, 192)
) -> List[Tuple[int, int, int]]:
    """Complete pipeline to get K crop centers from saliency map.
    
    Args:
        saliency: Saliency tensor (1, D, H, W) or (D, H, W)
        K: Number of crops to extract
        nms_distance: NMS distance threshold
        top_p: Top percentile for peak detection
        volume_shape: Volume dimensions for random fallback
        
    Returns:
        List of K (z, y, x) coordinates
    """
    # Convert to numpy
    if isinstance(saliency, torch.Tensor):
        saliency = saliency.detach().cpu().numpy()
    
    if saliency.ndim == 4:
        saliency = saliency[0]  # Remove batch dim
    
    # Get peaks
    coords = peaks_from_saliency(saliency, top_p=top_p, min_distance=8)
    
    if len(coords) == 0:
        # Fallback to random
        return random_diverse_centers(volume_shape, K)
    
    # Apply NMS
    scores = [float(saliency[c]) for c in coords]
    coords = nms_3d(coords, scores, distance=nms_distance)
    
    # Take top K
    coords = coords[:K]
    
    # Fill remaining with random if needed
    if len(coords) < K:
        remaining = K - len(coords)
        random_coords = random_diverse_centers(volume_shape, remaining)
        coords.extend(random_coords)
    
    return coords[:K]


def visualize_crops_on_volume(
    volume: np.ndarray,
    centers: List[Tuple[int, int, int]],
    crop_size: int = 96
) -> np.ndarray:
    """Visualize crop locations on mid-slices of volume.
    
    Args:
        volume: 3D volume (D, H, W)
        centers: List of crop centers
        crop_size: Size of crops
        
    Returns:
        RGB image of mid-slice with crop boxes
    """
    # Get mid-slice
    mid_z = volume.shape[0] // 2
    slice_img = volume[mid_z]
    
    # Convert to RGB
    slice_img = (slice_img * 255).astype(np.uint8)
    rgb = np.stack([slice_img] * 3, axis=-1)
    
    half = crop_size // 2
    
    # Draw boxes for crops that intersect this slice
    for cz, cy, cx in centers:
        if abs(cz - mid_z) < half:
            # Draw rectangle
            y1 = max(0, cy - half)
            y2 = min(volume.shape[1], cy + half)
            x1 = max(0, cx - half)
            x2 = min(volume.shape[2], cx + half)
            
            # Red border
            rgb[y1:y2, x1, 0] = 255
            rgb[y1:y2, x2-1, 0] = 255
            rgb[y1, x1:x2, 0] = 255
            rgb[y2-1, x1:x2, 0] = 255
    
    return rgb
