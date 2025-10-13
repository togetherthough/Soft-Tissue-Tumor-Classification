"""Visualization utilities for HieraCascade-STS"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import nibabel as nib
from pathlib import Path
from typing import List, Tuple, Optional
import torch


def save_saliency_overlay(
    volume: np.ndarray,
    saliency: np.ndarray,
    output_path: str,
    slice_axis: int = 0,
    slice_idx: Optional[int] = None,
    alpha: float = 0.5,
    cmap: str = 'jet'
):
    """Save saliency map overlay on volume slice
    
    Args:
        volume: 3D volume (D, H, W)
        saliency: 3D saliency map (D, H, W)
        output_path: Output image path
        slice_axis: 0, 1, or 2 for axial, coronal, sagittal
        slice_idx: Slice index (None for middle)
        alpha: Overlay transparency
        cmap: Colormap for saliency
    """
    # Get middle slice if not specified
    if slice_idx is None:
        slice_idx = volume.shape[slice_axis] // 2
    
    # Extract slices
    if slice_axis == 0:
        vol_slice = volume[slice_idx, :, :]
        sal_slice = saliency[slice_idx, :, :]
    elif slice_axis == 1:
        vol_slice = volume[:, slice_idx, :]
        sal_slice = saliency[:, slice_idx, :]
    else:
        vol_slice = volume[:, :, slice_idx]
        sal_slice = saliency[:, :, slice_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original volume
    axes[0].imshow(vol_slice, cmap='gray')
    axes[0].set_title('Original Volume')
    axes[0].axis('off')
    
    # Saliency map
    im1 = axes[1].imshow(sal_slice, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title('Saliency Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Overlay
    axes[2].imshow(vol_slice, cmap='gray')
    axes[2].imshow(sal_slice, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_saliency_volume(
    saliency: np.ndarray,
    output_path: str,
    affine: Optional[np.ndarray] = None
):
    """Save saliency map as NIfTI volume
    
    Args:
        saliency: 3D saliency map (D, H, W)
        output_path: Output .nii.gz path
        affine: Optional affine transformation
    """
    if affine is None:
        affine = np.eye(4)
    
    nii = nib.Nifti1Image(saliency, affine)
    nib.save(nii, output_path)


def visualize_crop_locations(
    volume: np.ndarray,
    centers: List[Tuple[int, int, int]],
    crop_size: int = 96,
    output_path: Optional[str] = None
) -> np.ndarray:
    """Visualize crop locations on volume slices
    
    Args:
        volume: 3D volume (D, H, W)
        centers: List of crop centers (z, y, x)
        crop_size: Crop size
        output_path: Optional path to save figure
        
    Returns:
        RGB image array
    """
    # Get three orthogonal slices
    mid_z, mid_y, mid_x = [s // 2 for s in volume.shape]
    
    slices = [
        volume[mid_z, :, :],  # Axial
        volume[:, mid_y, :],  # Coronal
        volume[:, :, mid_x],  # Sagittal
    ]
    
    titles = ['Axial', 'Coronal', 'Sagittal']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    half = crop_size // 2
    
    for i, (slice_img, title) in enumerate(zip(slices, titles)):
        # Display slice
        axes[i].imshow(slice_img, cmap='gray')
        
        # Draw crop boxes
        for cz, cy, cx in centers:
            if i == 0:  # Axial
                if abs(cz - mid_z) < half:
                    rect = plt.Rectangle(
                        (cx - half, cy - half),
                        crop_size, crop_size,
                        fill=False, edgecolor='red', linewidth=2
                    )
                    axes[i].add_patch(rect)
            elif i == 1:  # Coronal
                if abs(cy - mid_y) < half:
                    rect = plt.Rectangle(
                        (cx - half, cz - half),
                        crop_size, crop_size,
                        fill=False, edgecolor='red', linewidth=2
                    )
                    axes[i].add_patch(rect)
            else:  # Sagittal
                if abs(cx - mid_x) < half:
                    rect = plt.Rectangle(
                        (cy - half, cz - half),
                        crop_size, crop_size,
                        fill=False, edgecolor='red', linewidth=2
                    )
                    axes[i].add_patch(rect)
        
        axes[i].set_title(f'{title} (K={len(centers)} crops)')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
    normalize: bool = True,
    title: str = 'Confusion Matrix'
):
    """Plot confusion matrix
    
    Args:
        cm: Confusion matrix (C, C)
        class_names: List of class names
        output_path: Optional output path
        normalize: Normalize by true labels
        title: Plot title
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Ticks
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    train_metrics: List[Dict[str, float]],
    val_metrics: List[Dict[str, float]],
    output_path: Optional[str] = None
):
    """Plot training and validation curves
    
    Args:
        train_metrics: List of training metrics per epoch
        val_metrics: List of validation metrics per epoch
        output_path: Optional output path
    """
    epochs = range(1, len(train_metrics) + 1)
    
    # Determine metrics to plot
    metric_keys = set()
    for m in train_metrics + val_metrics:
        metric_keys.update(m.keys())
    
    # Remove loss-related for separate subplot
    plot_keys = [k for k in metric_keys if k != 'loss' and not k.endswith('_loss')]
    
    n_plots = 1 + len(plot_keys)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    # Plot loss
    if 'loss' in train_metrics[0]:
        train_loss = [m['loss'] for m in train_metrics]
        val_loss = [m['loss'] for m in val_metrics]
        
        axes[0].plot(epochs, train_loss, 'b-', label='Train')
        axes[0].plot(epochs, val_loss, 'r-', label='Val')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot other metrics
    for i, key in enumerate(plot_keys, start=1):
        train_vals = [m.get(key, np.nan) for m in train_metrics]
        val_vals = [m.get(key, np.nan) for m in val_metrics]
        
        axes[i].plot(epochs, train_vals, 'b-', label='Train')
        axes[i].plot(epochs, val_vals, 'r-', label='Val')
        axes[i].set_title(key.replace('_', ' ').title())
        axes[i].set_xlabel('Epoch')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(plot_keys) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_prediction_summary(
    study_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    output_path: str
):
    """Create CSV summary of predictions
    
    Args:
        study_ids: List of study IDs
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (N, C)
        class_names: Class names
        output_path: Output CSV path
    """
    import pandas as pd
    
    data = {
        'study_id': study_ids,
        'true_label': [class_names[y] for y in y_true],
        'pred_label': [class_names[y] for y in y_pred],
        'correct': y_true == y_pred
    }
    
    # Add probabilities
    for i, class_name in enumerate(class_names):
        data[f'prob_{class_name}'] = y_prob[:, i]
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"Saved prediction summary to {output_path}")
