"""Data preprocessing for HieraCascade-STS"""

import numpy as np
import nibabel as nib
from scipy import ndimage
from typing import Tuple, Optional
import torch


def load_nifti_volume(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI volume and return data + spacing.
    
    Args:
        path: Path to .nii or .nii.gz file
        
    Returns:
        volume: numpy array (D, H, W)
        spacing: voxel spacing in mm (dz, dy, dx)
    """
    nii = nib.load(path)
    volume = nii.get_fdata().astype(np.float32)
    
    # Get voxel spacing from header
    spacing = np.array(nii.header.get_zooms()[:3])
    
    return volume, spacing


def resample_to_isotropic(
    volume: np.ndarray,
    original_spacing: np.ndarray,
    target_spacing: float = 1.5,
    order: int = 1
) -> np.ndarray:
    """Resample volume to isotropic spacing.
    
    Args:
        volume: Input volume (D, H, W)
        original_spacing: Original voxel spacing (dz, dy, dx) in mm
        target_spacing: Target isotropic spacing in mm
        order: Interpolation order (1=linear, 3=cubic)
        
    Returns:
        Resampled volume
    """
    # Calculate zoom factors
    zoom_factors = original_spacing / target_spacing
    
    # Resample
    resampled = ndimage.zoom(volume, zoom_factors, order=order, mode='nearest')
    
    return resampled


def pad_or_crop_to_size(
    volume: np.ndarray,
    target_size: Tuple[int, int, int] = (192, 192, 192)
) -> np.ndarray:
    """Pad or center-crop volume to target size.
    
    Args:
        volume: Input volume (D, H, W)
        target_size: Target dimensions (D, H, W)
        
    Returns:
        Padded/cropped volume
    """
    current_size = np.array(volume.shape)
    target_size = np.array(target_size)
    
    # Calculate padding/cropping
    diff = target_size - current_size
    
    # Pad if needed
    pad_before = np.maximum(0, diff // 2)
    pad_after = np.maximum(0, diff - pad_before)
    padding = list(zip(pad_before, pad_after))
    
    if np.any(diff > 0):
        volume = np.pad(volume, padding, mode='constant', constant_values=0)
    
    # Crop if needed
    current_size = np.array(volume.shape)
    diff = current_size - target_size
    
    if np.any(diff > 0):
        crop_before = diff // 2
        crop_after = current_size - target_size - crop_before
        
        slices = tuple(slice(cb, -ca if ca > 0 else None) 
                      for cb, ca in zip(crop_before, crop_after))
        volume = volume[slices]
    
    return volume


def normalize_ct(volume: np.ndarray, hu_min: float = -150, hu_max: float = 250) -> np.ndarray:
    """Normalize CT volume (HU windowing).
    
    Args:
        volume: CT volume in Hounsfield Units
        hu_min: Minimum HU for windowing
        hu_max: Maximum HU for windowing
        
    Returns:
        Normalized volume in [0, 1]
    """
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)
    return volume.astype(np.float32)


def normalize_mri(volume: np.ndarray, clip_percentiles: Tuple[float, float] = (2, 98)) -> np.ndarray:
    """Normalize MRI volume (percentile clipping + z-score).
    
    Args:
        volume: MRI volume
        clip_percentiles: (lower, upper) percentiles for clipping
        
    Returns:
        Normalized volume
    """
    # Percentile clipping
    p_low, p_high = np.percentile(volume, clip_percentiles)
    volume = np.clip(volume, p_low, p_high)
    
    # Z-score normalization
    mean = np.mean(volume)
    std = np.std(volume)
    
    if std > 0:
        volume = (volume - mean) / std
    
    # Shift to [0, 1] range approximately
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    
    return volume.astype(np.float32)


def preprocess_volume(
    path: str,
    modality: str,
    target_spacing: float = 1.5,
    target_size: Tuple[int, int, int] = (192, 192, 192),
    ct_window: Tuple[float, float] = (-150, 250),
    mri_percentiles: Tuple[float, float] = (2, 98)
) -> np.ndarray:
    """Complete preprocessing pipeline for a volume.
    
    Args:
        path: Path to volume file
        modality: 'CT' or 'MRI'
        target_spacing: Target isotropic spacing in mm
        target_size: Target volume dimensions
        ct_window: (min, max) HU for CT windowing
        mri_percentiles: (lower, upper) percentiles for MRI clipping
        
    Returns:
        Preprocessed volume (D, H, W) in [0, 1]
    """
    # Load
    volume, spacing = load_nifti_volume(path)
    
    # Resample to isotropic
    volume = resample_to_isotropic(volume, spacing, target_spacing)
    
    # Pad/crop to target size
    volume = pad_or_crop_to_size(volume, target_size)
    
    # Intensity normalization
    if modality.upper() == 'CT':
        volume = normalize_ct(volume, ct_window[0], ct_window[1])
    else:  # MRI
        volume = normalize_mri(volume, mri_percentiles)
    
    return volume


def extract_crop(
    volume: np.ndarray,
    center: Tuple[int, int, int],
    crop_size: int = 96
) -> np.ndarray:
    """Extract a cubic crop from volume.
    
    Args:
        volume: Full volume (D, H, W)
        center: Center coordinates (z, y, x)
        crop_size: Edge length of crop cube
        
    Returns:
        Cropped volume (crop_size, crop_size, crop_size)
    """
    cz, cy, cx = center
    half = crop_size // 2
    
    # Calculate bounds with padding if near edges
    z_start = max(0, cz - half)
    z_end = min(volume.shape[0], cz + half)
    y_start = max(0, cy - half)
    y_end = min(volume.shape[1], cy + half)
    x_start = max(0, cx - half)
    x_end = min(volume.shape[2], cx + half)
    
    # Extract crop
    crop = volume[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Pad if near boundary
    if crop.shape != (crop_size, crop_size, crop_size):
        pad_z = (crop_size - crop.shape[0]) / 2
        pad_y = (crop_size - crop.shape[1]) / 2
        pad_x = (crop_size - crop.shape[2]) / 2
        
        padding = [
            (int(np.floor(pad_z)), int(np.ceil(pad_z))),
            (int(np.floor(pad_y)), int(np.ceil(pad_y))),
            (int(np.floor(pad_x)), int(np.ceil(pad_x)))
        ]
        
        crop = np.pad(crop, padding, mode='constant', constant_values=0)
    
    return crop


def augment_volume_3d(
    volume: np.ndarray,
    rotation_deg: float = 10,
    elastic: bool = True,
    gamma_range: Tuple[float, float] = (0.8, 1.2),
    noise_std: float = 0.01,
    blur_sigma: float = 0.5
) -> np.ndarray:
    """Apply 3D augmentations to volume.
    
    Args:
        volume: Input volume (D, H, W)
        rotation_deg: Maximum rotation in degrees
        elastic: Apply elastic deformation
        gamma_range: Range for gamma correction
        noise_std: Std dev for Gaussian noise
        blur_sigma: Sigma for Gaussian blur
        
    Returns:
        Augmented volume
    """
    # Random rotation
    if rotation_deg > 0:
        angle = np.random.uniform(-rotation_deg, rotation_deg)
        axes = [(0, 1), (0, 2), (1, 2)]
        plane = axes[np.random.randint(3)]
        volume = ndimage.rotate(volume, angle, axes=plane, reshape=False, order=1, mode='nearest')
    
    # Elastic deformation (simplified)
    if elastic and np.random.rand() > 0.5:
        alpha = np.random.uniform(10, 30)
        sigma = np.random.uniform(3, 5)
        
        shape = volume.shape
        dz = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dx = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        
        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(z + dz, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        volume = ndimage.map_coordinates(volume, indices, order=1, mode='reflect').reshape(shape)
    
    # Gamma correction
    if gamma_range:
        gamma = np.random.uniform(*gamma_range)
        volume = np.power(volume, gamma)
    
    # Gaussian noise
    if noise_std > 0 and np.random.rand() > 0.5:
        noise = np.random.normal(0, noise_std, volume.shape)
        volume = volume + noise
        volume = np.clip(volume, 0, 1)
    
    # Gaussian blur
    if blur_sigma > 0 and np.random.rand() > 0.5:
        sigma = np.random.uniform(0, blur_sigma)
        volume = ndimage.gaussian_filter(volume, sigma)
    
    return volume.astype(np.float32)
