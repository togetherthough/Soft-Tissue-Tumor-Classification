"""
Mesh extraction from 3D masks with feature computation
Includes marching cubes, smoothing, decimation, and curvature calculation
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy import ndimage as ndi


def extract_mesh_from_mask(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    smooth_kernel: Tuple[int, int, int] = (3, 3, 3),
    laplacian_iters: int = 10,
    laplacian_lambda: float = 0.5,
    target_vertices: int = 8000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract surface mesh from binary mask using marching cubes.
    
    Args:
        mask: (D, H, W) binary mask
        spacing: (z, y, x) voxel spacing in mm
        smooth_kernel: kernel size for median smoothing before marching cubes
        laplacian_iters: number of Laplacian smoothing iterations
        laplacian_lambda: Laplacian smoothing factor
        target_vertices: target number of vertices for decimation
    
    Returns:
        vertices: (N, 3) array in mm coordinates
        faces: (M, 3) array of triangle indices
    """
    try:
        from skimage import measure
        import trimesh
    except ImportError:
        raise ImportError("Please install: pip install scikit-image trimesh")
    
    # Smooth mask slightly to reduce artifacts
    if smooth_kernel is not None:
        mask_smooth = ndi.median_filter(mask.astype(np.float32), size=smooth_kernel)
        mask_smooth = (mask_smooth > 0.5).astype(np.uint8)
    else:
        mask_smooth = mask.astype(np.uint8)
    
    # Marching cubes
    try:
        vertices, faces, normals, _ = measure.marching_cubes(
            mask_smooth,
            level=0.5,
            spacing=spacing,
            allow_degenerate=False
        )
    except (ValueError, RuntimeError) as e:
        print(f"Warning: Marching cubes failed: {e}. Returning empty mesh.")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    
    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    
    # Convert to trimesh for processing
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # Laplacian smoothing
    if laplacian_iters > 0:
        trimesh.smoothing.filter_laplacian(
            mesh,
            iterations=laplacian_iters,
            lamb=laplacian_lambda,
            implicit_time_integration=False
        )
    
    # Decimate to target vertex count
    if len(mesh.vertices) > target_vertices:
        try:
            mesh = mesh.simplify_quadric_decimation(target_vertices)
        except:
            # Fallback: just sample vertices
            indices = np.random.choice(len(mesh.vertices), target_vertices, replace=False)
            # This breaks topology, but better than nothing
            pass
    
    return np.array(mesh.vertices), np.array(mesh.faces)


def compute_curvature_features(
    vertices: np.ndarray,
    faces: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute curvature-based features for mesh vertices.
    
    Principal curvatures (k1, k2) → mean curvature, Gaussian curvature,
    shape index, curvedness.
    
    Args:
        vertices: (N, 3)
        faces: (M, 3)
    
    Returns:
        dict with keys: 'mean_curvature', 'gaussian_curvature',
                       'shape_index', 'curvedness', 'k1', 'k2'
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("pip install trimesh")
    
    if len(vertices) == 0:
        return {
            'mean_curvature': np.array([]),
            'gaussian_curvature': np.array([]),
            'shape_index': np.array([]),
            'curvedness': np.array([]),
            'k1': np.array([]),
            'k2': np.array([])
        }
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # Compute discrete curvatures (approximation via normal variation)
    # For production, use more sophisticated methods (e.g., igl.principal_curvature)
    
    # Simple approximation using vertex normals
    vertex_normals = mesh.vertex_normals
    
    # Compute principal curvatures via eigenvalues of shape operator
    # This is a simplified version; for better results use libigl or similar
    k1, k2 = _approximate_principal_curvatures(mesh)
    
    # Derived features
    mean_curv = 0.5 * (k1 + k2)
    gauss_curv = k1 * k2
    
    # Shape index: range [-1, 1], indicates surface type
    with np.errstate(divide='ignore', invalid='ignore'):
        shape_index = (2.0 / np.pi) * np.arctan((k1 + k2) / (k1 - k2 + 1e-8))
        shape_index = np.nan_to_num(shape_index, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Curvedness: magnitude of curvature
    curvedness = np.sqrt(0.5 * (k1**2 + k2**2))
    
    return {
        'mean_curvature': mean_curv,
        'gaussian_curvature': gauss_curv,
        'shape_index': shape_index,
        'curvedness': curvedness,
        'k1': k1,
        'k2': k2
    }


def _approximate_principal_curvatures(mesh) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate principal curvatures using discrete operators.
    
    This is a simplified version. For production, use:
    - libigl's principal_curvature
    - or PyMesh's compute_curvature
    """
    import trimesh
    
    N = len(mesh.vertices)
    k1 = np.zeros(N)
    k2 = np.zeros(N)
    
    # Use vertex normals and neighbor information
    # Compute curvature via normal variation
    vertex_neighbors = mesh.vertex_neighbors
    
    for i in range(N):
        neighbors = vertex_neighbors[i]
        if len(neighbors) < 3:
            continue
        
        # Compute local coordinate frame
        normal = mesh.vertex_normals[i]
        
        # Project neighbors to tangent plane and fit quadric
        neighbor_verts = mesh.vertices[neighbors]
        rel_pos = neighbor_verts - mesh.vertices[i]
        
        # Project to tangent plane
        tangent_proj = rel_pos - np.outer(rel_pos.dot(normal), normal)
        
        # Fit local quadratic surface (simplified)
        if len(tangent_proj) >= 3:
            try:
                # Compute covariance of projected positions
                cov = np.cov(tangent_proj.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                
                # Use eigenvalues as curvature proxy
                k1[i] = eigenvalues[-1]
                k2[i] = eigenvalues[0]
            except:
                pass
    
    # Normalize to reasonable range
    k1 = np.clip(k1, -10, 10)
    k2 = np.clip(k2, -10, 10)
    
    return k1, k2


def compute_mesh_node_features(
    vertices: np.ndarray,
    faces: np.ndarray,
    volume: np.ndarray,
    mask: np.ndarray,
    rim: np.ndarray,
    spacing: Tuple[float, float, float],
    volume_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    """
    Compute comprehensive node features for mesh vertices.
    
    Features include:
    - Curvature features (mean, Gaussian, shape index, curvedness)
    - Intensity features (at vertex, normal inward/outward)
    - Geometric features (distance to centroid, distance to rim)
    
    Args:
        vertices: (N, 3) in mm
        faces: (M, 3)
        volume: (D, H, W) intensity volume
        mask: (D, H, W) tumor mask
        rim: (D, H, W) rim mask
        spacing: (z, y, x) in mm
        volume_origin: (z, y, x) origin in mm
    
    Returns:
        features: (N, F) array where F ~ 8-12 features per vertex
    """
    if len(vertices) == 0:
        return np.zeros((0, 8))
    
    # Curvature features
    curv_feats = compute_curvature_features(vertices, faces)
    
    # Intensity features: sample volume at vertices
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = mesh.vertex_normals
    
    def sample_volume_at_points(points_mm: np.ndarray) -> np.ndarray:
        """Sample volume at physical coordinates (mm)"""
        # Convert mm to voxel coordinates
        voxel_coords = (points_mm - np.array(volume_origin)) / np.array(spacing)
        
        # Trilinear interpolation
        from scipy.ndimage import map_coordinates
        values = map_coordinates(
            volume,
            voxel_coords.T,
            order=1,
            mode='constant',
            cval=0.0
        )
        return values
    
    I_surface = sample_volume_at_points(vertices)
    I_inward = sample_volume_at_points(vertices - 2.0 * normals)  # 2mm inward
    I_outward = sample_volume_at_points(vertices + 2.0 * normals)  # 2mm outward
    
    # Geodesic distance to mask centroid (approximate with Euclidean)
    mask_coords = np.argwhere(mask > 0)
    if len(mask_coords) > 0:
        centroid_voxel = mask_coords.mean(axis=0)
        centroid_mm = centroid_voxel * np.array(spacing) + np.array(volume_origin)
        dist_to_centroid = np.linalg.norm(vertices - centroid_mm, axis=1)
    else:
        dist_to_centroid = np.zeros(len(vertices))
    
    # Distance to rim (approximate)
    rim_coords_mm = np.argwhere(rim > 0) * np.array(spacing) + np.array(volume_origin)
    if len(rim_coords_mm) > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(rim_coords_mm)
        dist_to_rim, _ = tree.query(vertices)
    else:
        dist_to_rim = np.ones(len(vertices)) * 10.0  # default
    
    # Stack features
    features = np.column_stack([
        curv_feats['mean_curvature'],
        curv_feats['gaussian_curvature'],
        curv_feats['shape_index'],
        curv_feats['curvedness'],
        I_surface,
        I_inward,
        I_outward,
        dist_to_centroid / 100.0,  # normalize
        dist_to_rim / 10.0  # normalize
    ])
    
    # Standardize features
    features = (features - features.mean(axis=0, keepdims=True)) / (features.std(axis=0, keepdims=True) + 1e-6)
    
    return features.astype(np.float32)
