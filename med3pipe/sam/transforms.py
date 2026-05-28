"""
SAM-Med3D preprocessing transforms.

This module provides transforms for preparing 3D medical images for SAM-Med3D,
using a resize-then-pad approach that minimizes data loss.
"""

import torchio as tio
import SimpleITK as sitk
import torch
from pathlib import Path


class ResizeLargestTo(tio.Transform):
    """Resize volume so that the largest dimension becomes target_size.
    
    This preserves aspect ratio and minimizes data loss compared to CropOrPad.
    Useful for preparing data for SAM-Med3D where we want 128^3 format.
    
    Args:
        target_size: Target size for the largest dimension (default: 128)
    
    Example:
        >>> transform = ResizeLargestTo(target_size=128)
        >>> # Volume with shape (1, 200, 150, 100) becomes (1, 128, 96, 64)
    """
    def __init__(self, target_size: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.target_size = target_size
    
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # Get the first image to determine spatial shape
        first_image = subject.get_first_image()
        spatial_shape = first_image.spatial_shape  # (D, H, W)
        
        # Find largest dimension
        max_dim = max(spatial_shape)
        
        # Calculate scale factor to make largest dimension = target_size
        if max_dim > 0:
            scale = self.target_size / max_dim
        else:
            scale = 1.0
        
        # Calculate new shape (all dimensions scaled proportionally)
        new_shape = tuple(int(dim * scale) for dim in spatial_shape)
        
        # Apply resize to all images in subject
        resize_transform = tio.Resize(target_shape=new_shape)
        subject = resize_transform(subject)
        
        return subject


def make_sam3d_transform(img_size: int = 128, normalize: bool = False) -> tio.Compose:
    """Create preprocessing transform pipeline for SAM-Med3D.
    
    Uses resize-then-pad approach to minimize data loss:
    1. ToCanonical: Reorient to standard RAS+ orientation
    2. ResizeLargestTo: Resize so largest dimension becomes img_size (preserves aspect ratio)
    3. CropOrPad: Pad remaining dimensions to img_size^3 (minimal padding needed)
    4. ZNormalization: Optional normalization (disabled by default for visualization)
    
    Args:
        img_size: Target cube size (default: 128)
        normalize: Whether to apply Z-normalization (default: False)
    
    Returns:
        Composed transform pipeline
    """
    transforms = [
        tio.ToCanonical(),
        ResizeLargestTo(target_size=img_size),
        tio.CropOrPad(target_shape=(img_size, img_size, img_size)),
    ]
    
    if normalize:
        def _znorm_masking_method(x):
            return x > 0
        transforms.append(tio.ZNormalization(masking_method=_znorm_masking_method))
    
    return tio.Compose(transforms)


def load_volume_resize_pad(img_path: Path, img_size: int = 128, normalize: bool = False) -> torch.Tensor:
    """Load a volume with resize-then-pad preprocessing.
    
    This function minimizes data loss by:
    1. Resizing so the largest dimension becomes img_size
    2. Padding the smaller dimensions to create img_size^3 cube
    
    Args:
        img_path: Path to NIfTI image file
        img_size: Target cube size (default: 128)
        normalize: Whether to apply Z-normalization (default: False)
    
    Returns:
        Tensor of shape (1, 1, img_size, img_size, img_size)
    """
    sitk_img = sitk.ReadImage(str(img_path))
    sitk_arr, _ = tio.data.io.sitk_to_nib(sitk_img)
    subject = tio.Subject(image=tio.ScalarImage(tensor=sitk_arr))
    
    transform = make_sam3d_transform(img_size=img_size, normalize=normalize)
    subject = transform(subject)
    
    image = subject.image.data.clone().detach()
    image = image.unsqueeze(0)  # Add batch dimension: (1, 1, D, H, W)
    image = image.float()
    return image


def load_mask_resize_pad(mask_path: Path, img_size: int = 128) -> torch.Tensor:
    """Load a mask with resize-then-pad preprocessing.
    
    Args:
        mask_path: Path to NIfTI mask file
        img_size: Target cube size (default: 128)
    
    Returns:
        Binary mask array of shape (img_size, img_size, img_size)
    """
    sitk_mask = sitk.ReadImage(str(mask_path))
    mask_arr, _ = tio.data.io.sitk_to_nib(sitk_mask)
    subject = tio.Subject(label=tio.LabelMap(tensor=mask_arr))
    
    transform = tio.Compose([
        tio.ToCanonical(),
        ResizeLargestTo(target_size=img_size),
        tio.CropOrPad(target_shape=(img_size, img_size, img_size)),
    ])
    subject = transform(subject)
    
    mask = subject.label.data.squeeze().numpy()
    mask = (mask > 0).astype(float)
    return mask
