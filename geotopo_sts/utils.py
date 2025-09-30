"""
Utility functions for GeoTopo-STS
Helper functions for visualization, logging, and data handling
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import torch
from pathlib import Path


def visualize_sample(
    volume: np.ndarray,
    mask: np.ndarray,
    rim: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Visualize a 3D sample with central slices.
    
    Args:
        volume: (D, H, W) intensity volume
        mask: (D, H, W) tumor mask
        rim: (D, H, W) rim mask
        save_path: optional path to save figure
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    # Get central slices
    d, h, w = volume.shape
    slices = {
        'axial': d // 2,
        'coronal': h // 2,
        'sagittal': w // 2
    }
    
    for col, (plane, idx) in enumerate(slices.items()):
        if plane == 'axial':
            vol_slice = volume[idx, :, :]
            mask_slice = mask[idx, :, :]
            rim_slice = rim[idx, :, :]
        elif plane == 'coronal':
            vol_slice = volume[:, idx, :]
            mask_slice = mask[:, idx, :]
            rim_slice = rim[:, idx, :]
        else:  # sagittal
            vol_slice = volume[:, :, idx]
            mask_slice = mask[:, :, idx]
            rim_slice = rim[:, :, idx]
        
        # Volume
        axes[0, col].imshow(vol_slice, cmap='gray')
        axes[0, col].set_title(f'{plane.capitalize()} - Volume')
        axes[0, col].axis('off')
        
        # Mask overlay
        axes[1, col].imshow(vol_slice, cmap='gray')
        axes[1, col].imshow(mask_slice, cmap='Reds', alpha=0.3 * (mask_slice > 0))
        axes[1, col].set_title(f'{plane.capitalize()} - Tumor')
        axes[1, col].axis('off')
        
        # Rim overlay
        axes[2, col].imshow(vol_slice, cmap='gray')
        axes[2, col].imshow(rim_slice, cmap='Blues', alpha=0.3 * (rim_slice > 0))
        axes[2, col].set_title(f'{plane.capitalize()} - Rim')
        axes[2, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    features: Optional[np.ndarray] = None,
    feature_name: str = 'curvature',
    save_path: Optional[str] = None
):
    """
    Visualize 3D mesh with optional feature coloring.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) triangle faces
        features: (N,) per-vertex features for coloring
        feature_name: name of feature being visualized
        save_path: optional path to save figure
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("3D plotting not available")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh collection
    mesh_triangles = vertices[faces]
    
    if features is not None:
        # Color by features
        face_colors = features[faces].mean(axis=1)
        collection = Poly3DCollection(
            mesh_triangles,
            alpha=0.7,
            edgecolor='k',
            linewidths=0.1
        )
        collection.set_array(face_colors)
        collection.set_cmap('viridis')
        ax.add_collection3d(collection)
        plt.colorbar(collection, ax=ax, label=feature_name, shrink=0.6)
    else:
        # Uniform color
        collection = Poly3DCollection(
            mesh_triangles,
            alpha=0.7,
            facecolor='steelblue',
            edgecolor='k',
            linewidths=0.1
        )
        ax.add_collection3d(collection)
    
    # Set limits
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Tumor Surface Mesh')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_persistence_diagram(
    diagram: np.ndarray,
    dim: int = 0,
    save_path: Optional[str] = None
):
    """
    Plot persistence diagram (birth-death plot).
    
    Args:
        diagram: (n, 2) array with [birth, death] times
        dim: homology dimension
        save_path: optional path to save
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    
    # Filter infinite persistence
    finite_mask = deaths < np.inf
    births_finite = births[finite_mask]
    deaths_finite = deaths[finite_mask]
    
    # Plot points
    ax.scatter(births_finite, deaths_finite, alpha=0.6, s=30, label=f'H{dim}')
    
    # Diagonal line (birth = death)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='birth = death')
    
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(f'Persistence Diagram (H{dim})')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    history: Dict[str, List[Dict]],
    metrics: List[str] = ['loss', 'macro_f1'],
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves.
    
    Args:
        history: dict with 'train' and 'val' lists of metric dicts
        metrics: list of metric names to plot
        save_path: optional save path
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        train_values = [epoch.get(metric, np.nan) for epoch in history['train']]
        val_values = [epoch.get(metric, np.nan) for epoch in history['val']]
        
        epochs = range(1, len(train_values) + 1)
        
        axes[idx].plot(epochs, train_values, label='Train', marker='o', markersize=3)
        axes[idx].plot(epochs, val_values, label='Val', marker='s', markersize=3)
        
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].set_title(f'{metric.replace("_", " ").title()} Curves')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_embeddings_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize embeddings using UMAP.
    
    Args:
        embeddings: (N, D) embedding vectors
        labels: (N,) class labels
        class_names: optional list of class names
        save_path: optional save path
    """
    try:
        from umap import UMAP
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return
    
    # Reduce to 2D
    reducer = UMAP(n_components=2, random_state=42)
    embedded_2d = reducer.fit_transform(embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        embedded_2d[:, 0],
        embedded_2d[:, 1],
        c=labels,
        cmap='tab20',
        alpha=0.6,
        s=20
    )
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Embedding Visualization (UMAP)')
    
    # Legend
    if class_names is not None:
        unique_labels = np.unique(labels)
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=plt.cm.tab20(l / len(class_names)), 
                             markersize=8, label=class_names[l])
                  for l in unique_labels if l < len(class_names)]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_class_distribution(labels: np.ndarray) -> Dict[int, int]:
    """Compute class distribution from labels"""
    unique, counts = np.unique(labels, return_counts=True)
    return {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}


def print_model_summary(model: torch.nn.Module):
    """Print model architecture summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*60)
    
    # Print module sizes
    print("\nModule sizes:")
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:30s}: {n_params:>12,} params")
    print("="*60 + "\n")


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_predictions_with_confidence(
    case_ids: List[str],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    class_names: List[str],
    save_path: str
):
    """
    Save predictions with confidence scores to CSV.
    
    Args:
        case_ids: list of case identifiers
        predictions: (N,) predicted classes
        probabilities: (N, K) class probabilities
        true_labels: (N,) true classes
        class_names: list of class names
        save_path: output CSV path
    """
    import pandas as pd
    
    data = {
        'case_id': case_ids,
        'true_label': true_labels,
        'true_class': [class_names[l] if l < len(class_names) else f'Class_{l}' 
                       for l in true_labels],
        'predicted_label': predictions,
        'predicted_class': [class_names[p] if p < len(class_names) else f'Class_{p}' 
                           for p in predictions],
        'confidence': probabilities.max(axis=1),
        'correct': predictions == true_labels
    }
    
    # Add top-3 predictions
    top3_indices = np.argsort(-probabilities, axis=1)[:, :3]
    for k in range(3):
        data[f'top{k+1}_class'] = [class_names[idx[k]] if idx[k] < len(class_names) 
                                   else f'Class_{idx[k]}' 
                                   for idx in top3_indices]
        data[f'top{k+1}_prob'] = [probabilities[i, idx[k]] 
                                 for i, idx in enumerate(top3_indices)]
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Saved predictions to {save_path}")


def create_hierarchy_tree(
    ancestors: Dict[int, List[int]],
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Visualize class hierarchy as a tree.
    
    Args:
        ancestors: dict mapping class_id -> list of ancestor IDs
        class_names: list of class names
        save_path: optional save path for figure
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX not installed. Install with: pip install networkx")
        return
    
    G = nx.DiGraph()
    
    # Add nodes
    for i, name in enumerate(class_names):
        G.add_node(i, label=name)
    
    # Add edges (child -> parent)
    for child, parents in ancestors.items():
        for parent in parents:
            if parent < len(class_names):
                G.add_edge(parent, child)
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw
    fig, ax = plt.subplots(figsize=(14, 10))
    
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=15, ax=ax)
    
    labels = {i: class_names[i][:15] for i in range(len(class_names))}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title('Class Hierarchy Tree')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
