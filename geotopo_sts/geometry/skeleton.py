"""
Skeleton extraction from 3D masks
Optional component for skeleton graph pathway
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy import ndimage as ndi


def skeletonize_mask(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    method: str = 'medial_axis'
) -> np.ndarray:
    """
    Skeletonize a 3D binary mask.
    
    Args:
        mask: (D, H, W) binary mask
        spacing: (z, y, x) voxel spacing
        method: 'medial_axis' or 'thinning'
    
    Returns:
        skeleton: (D, H, W) binary skeleton
    """
    try:
        from skimage.morphology import skeletonize_3d, medial_axis
    except ImportError:
        raise ImportError("pip install scikit-image")
    
    if method == 'medial_axis':
        # Medial axis transform
        skeleton, distance = medial_axis(mask, return_distance=True)
    elif method == 'thinning':
        # Morphological thinning
        skeleton = skeletonize_3d(mask)
    else:
        raise ValueError(f"Unknown skeletonization method: {method}")
    
    return skeleton.astype(np.uint8)


def extract_skeleton_graph(
    skeleton: np.ndarray,
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    min_branch_length_mm: float = 6.0,
    merge_threshold_mm: float = 3.0
) -> Dict[str, np.ndarray]:
    """
    Extract skeleton graph from skeletonized mask.
    
    Converts skeleton voxels to nodes and edges, then prunes short branches
    and merges nearby nodes.
    
    Args:
        skeleton: (D, H, W) binary skeleton
        mask: (D, H, W) original tumor mask
        spacing: (z, y, x) voxel spacing in mm
        min_branch_length_mm: prune branches shorter than this
        merge_threshold_mm: merge nodes closer than this
    
    Returns:
        dict with:
            'nodes': (N, 3) positions in mm
            'features': (N, F) node features
            'edges': (E, 2) edge list
            'adjacency': (N, N) adjacency matrix
    """
    # Find skeleton voxels
    skel_coords = np.argwhere(skeleton > 0)
    
    if len(skel_coords) == 0:
        return {
            'nodes': np.zeros((0, 3)),
            'features': np.zeros((0, 4)),
            'edges': np.zeros((0, 2), dtype=np.int64),
            'adjacency': np.zeros((0, 0))
        }
    
    # Convert to mm
    nodes_mm = skel_coords * np.array(spacing)
    
    # Build adjacency by 26-connectivity in voxel space
    from scipy.spatial import cKDTree
    tree = cKDTree(skel_coords)
    
    # Find neighbors within sqrt(3) voxels (26-connected)
    adjacency_list = tree.query_ball_tree(tree, r=1.8)
    
    edges = []
    for i, neighbors in enumerate(adjacency_list):
        for j in neighbors:
            if j > i:  # avoid duplicates
                edges.append([i, j])
    
    edges = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)
    
    # Compute node features
    # 1. Distance to surface (via distance transform)
    distance_transform = ndi.distance_transform_edt(mask, sampling=spacing)
    radii = distance_transform[skeleton > 0]
    
    # 2. Geodesic distance to root (approximate with distance to centroid)
    centroid = skel_coords.mean(axis=0)
    dist_to_root = np.linalg.norm(skel_coords - centroid, axis=1)
    
    # 3. Degree (number of connections)
    degrees = np.array([len(neighbors) - 1 for neighbors in adjacency_list])  # -1 to exclude self
    
    # Stack features
    features = np.column_stack([
        radii / 10.0,  # normalized radius
        dist_to_root / 50.0,  # normalized distance
        degrees / 6.0,  # normalized degree
        np.ones(len(nodes_mm))  # placeholder
    ]).astype(np.float32)
    
    # TODO: Prune short branches and merge nearby nodes
    # For now, return raw skeleton graph
    
    # Build adjacency matrix
    N = len(nodes_mm)
    adjacency = np.zeros((N, N), dtype=np.uint8)
    if len(edges) > 0:
        adjacency[edges[:, 0], edges[:, 1]] = 1
        adjacency[edges[:, 1], edges[:, 0]] = 1
    
    return {
        'nodes': nodes_mm.astype(np.float32),
        'features': features,
        'edges': edges,
        'adjacency': adjacency
    }
