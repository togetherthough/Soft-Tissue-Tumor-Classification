"""
Preprocessing utilities for 3D medical images
Includes resampling, bias correction, intensity normalization, and cropping
"""

import numpy as np
import scipy.ndimage as ndi
from typing import Tuple, Optional, Dict, Any


def resample_volume(
    volume: np.ndarray,
    mask: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    order: int = 3
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """
    Resample 3D volume and mask to target isotropic spacing.
    
    Args:
        volume: (D, H, W) array
        mask: (D, H, W) binary mask
        original_spacing: (z, y, x) spacing in mm
        target_spacing: target isotropic spacing in mm
        order: interpolation order (3=cubic for volume, 0=nearest for mask)
    
    Returns:
        resampled_volume, resampled_mask, target_spacing
    """
    # Compute zoom factors
    zoom_factors = np.array(original_spacing) / np.array(target_spacing)
    
    # Resample volume
    volume_resampled = ndi.zoom(volume, zoom_factors, order=order, mode='constant', cval=volume.min())
    
    # Resample mask (nearest neighbor)
    mask_resampled = ndi.zoom(mask.astype(np.float32), zoom_factors, order=0, mode='constant', cval=0)
    mask_resampled = (mask_resampled > 0.5).astype(np.uint8)
    
    return volume_resampled, mask_resampled, target_spacing


def bias_field_correction(volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply N4 bias field correction (simplified version).
    For production, use SimpleITK's N4BiasFieldCorrectionImageFilter.
    
    This is a placeholder that applies a simple high-pass filter.
    """
    try:
        import SimpleITK as sitk
        
        # Convert to SimpleITK
        img = sitk.GetImageFromArray(volume.astype(np.float32))
        
        if mask is not None:
            mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
        else:
            # Create mask from non-zero voxels
            mask_img = sitk.GetImageFromArray((volume > volume.min()).astype(np.uint8))
        
        # N4 correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
        corrected = corrector.Execute(img, mask_img)
        
        return sitk.GetArrayFromImage(corrected)
    
    except ImportError:
        # Fallback: simple high-pass filter approximation
        print("Warning: SimpleITK not found, using approximation for bias correction")
        from scipy.ndimage import gaussian_filter
        
        smoothed = gaussian_filter(volume.astype(np.float32), sigma=25)
        corrected = volume / (smoothed + 1e-5) * smoothed.mean()
        
        return corrected


def normalize_intensity(
    volume: np.ndarray,
    modality: str = 'mri',
    mask: Optional[np.ndarray] = None,
    percentile_clip: Tuple[float, float] = (1, 99),
    window: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Normalize intensity values.
    
    Args:
        volume: (D, H, W) array
        modality: 'mri' or 'ct'
        mask: optional ROI mask for normalization
        percentile_clip: (low, high) percentiles for MRI
        window: (min, max) HU window for CT
    
    Returns:
        normalized volume (z-scored)
    """
    volume = volume.astype(np.float32)
    
    if modality.lower() == 'ct':
        # CT: apply HU window
        if window is None:
            window = (-150, 350)  # soft tissue window
        volume = np.clip(volume, window[0], window[1])
    
    elif modality.lower() == 'mri':
        # MRI: percentile clipping
        if mask is not None:
            roi_vals = volume[mask > 0]
        else:
            roi_vals = volume[volume > volume.min()]
        
        p_low, p_high = np.percentile(roi_vals, percentile_clip)
        volume = np.clip(volume, p_low, p_high)
    
    # Z-score normalization
    if mask is not None:
        roi_vals = volume[mask > 0]
    else:
        roi_vals = volume
    
    mean = roi_vals.mean()
    std = roi_vals.std()
    
    if std > 1e-6:
        volume = (volume - mean) / std
    else:
        volume = volume - mean
    
    return volume


def extract_crop_and_rim(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    rim_radius_mm: float = 10.0,
    padding_voxels: int = 25,
    target_size: Optional[Tuple[int, int, int]] = None
) -> Dict[str, np.ndarray]:
    """
    Extract tight crop around tumor with padding and create peritumoral rim.
    
    Args:
        volume: (D, H, W) normalized volume
        mask: (D, H, W) binary tumor mask
        spacing: (z, y, x) voxel spacing in mm
        rim_radius_mm: radius for peritumoral rim in mm
        padding_voxels: padding on each side of bounding box
        target_size: optional (D, H, W) to resize to
    
    Returns:
        dict with keys: 'volume', 'mask', 'rim', 'bbox'
    """
    # Find bounding box
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        raise ValueError("Empty mask provided")
    
    z_min, y_min, x_min = [c.min() for c in coords]
    z_max, y_max, x_max = [c.max() for c in coords]
    
    # Add padding
    D, H, W = volume.shape
    z_min = max(0, z_min - padding_voxels)
    y_min = max(0, y_min - padding_voxels)
    x_min = max(0, x_min - padding_voxels)
    z_max = min(D, z_max + padding_voxels + 1)
    y_max = min(H, y_max + padding_voxels + 1)
    x_max = min(W, x_max + padding_voxels + 1)
    
    # Crop
    volume_crop = volume[z_min:z_max, y_min:y_max, x_min:x_max].copy()
    mask_crop = mask[z_min:z_max, y_min:y_max, x_min:x_max].copy()
    
    # Create rim: dilate mask by rim_radius_mm, then subtract original
    rim_radius_voxels = rim_radius_mm / np.mean(spacing)
    
    # Dilate mask
    from scipy.ndimage import distance_transform_edt
    dt = distance_transform_edt(~mask_crop.astype(bool), sampling=spacing)
    dilated_mask = dt <= rim_radius_mm
    
    # Rim = dilated - original
    rim = (dilated_mask & ~mask_crop.astype(bool)).astype(np.uint8)
    
    # Optional: resize to target size
    if target_size is not None:
        zoom_factors = np.array(target_size) / np.array(volume_crop.shape)
        volume_crop = ndi.zoom(volume_crop, zoom_factors, order=3)
        mask_crop = ndi.zoom(mask_crop.astype(np.float32), zoom_factors, order=0)
        rim = ndi.zoom(rim.astype(np.float32), zoom_factors, order=0)
        
        mask_crop = (mask_crop > 0.5).astype(np.uint8)
        rim = (rim > 0.5).astype(np.uint8)
    
    return {
        'volume': volume_crop,
        'mask': mask_crop,
        'rim': rim,
        'bbox': (z_min, z_max, y_min, y_max, x_min, x_max)
    }


def preprocess_case(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    modality: str = 'mri',
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for one case.
    
    Args:
        volume: (D, H, W) raw volume
        mask: (D, H, W) binary mask
        spacing: (z, y, x) original spacing in mm
        modality: 'mri' or 'ct'
        config: preprocessing configuration dict
    
    Returns:
        dict with preprocessed data
    """
    if config is None:
        config = {
            'target_spacing': [1.5, 1.5, 1.5],
            'rim_radius_mm': 10.0,
            'crop_padding': 25,
            'crop_size': [160, 160, 160],
            'bias_correction': True,
            'percentile_clip': [1, 99],
            'window': [-150, 350] if modality == 'ct' else None
        }
    
    # Step 1: Resample
    volume, mask, new_spacing = resample_volume(
        volume, mask,
        original_spacing=spacing,
        target_spacing=tuple(config['target_spacing'])
    )
    
    # Step 2: Bias correction (MRI only)
    if modality.lower() == 'mri' and config.get('bias_correction', False):
        volume = bias_field_correction(volume, mask)
    
    # Step 3: Intensity normalization
    volume = normalize_intensity(
        volume,
        modality=modality,
        mask=mask,
        percentile_clip=tuple(config.get('percentile_clip', [1, 99])),
        window=config.get('window')
    )
    
    # Step 4: Crop and create rim
    cropped = extract_crop_and_rim(
        volume,
        mask,
        spacing=new_spacing,
        rim_radius_mm=config.get('rim_radius_mm', 10.0),
        padding_voxels=config.get('crop_padding', 25),
        target_size=tuple(config['crop_size']) if 'crop_size' in config else None
    )
    
    return {
        'volume': cropped['volume'],
        'mask': cropped['mask'],
        'rim': cropped['rim'],
        'spacing': new_spacing,
        'bbox': cropped['bbox']
    }
