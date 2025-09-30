"""
Persistent Homology feature extraction
Computes persistence diagrams and converts to persistence images
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import ndimage as ndi


def compute_persistence_diagrams(
    volume: np.ndarray,
    filtration_type: str = 'sublevel',
    dims: List[int] = [0, 1, 2],
    spacing: Optional[Tuple[float, float, float]] = None
) -> List[np.ndarray]:
    """
    Compute persistent homology diagrams using cubical complexes.
    
    Args:
        volume: (D, H, W) scalar field for filtration
        filtration_type: 'sublevel' or 'superlevel'
        dims: homology dimensions to compute (0=components, 1=loops, 2=voids)
        spacing: voxel spacing in mm (for physical units)
    
    Returns:
        List of diagrams, one per dimension. Each diagram is (n_features, 2)
        array with columns [birth, death]
    """
    try:
        # Try using gudhi (preferred) or giotto-tda
        import gudhi as gd
        use_gudhi = True
    except ImportError:
        try:
            from gtda.homology import CubicalPersistence
            use_gudhi = False
        except ImportError:
            raise ImportError(
                "Please install: pip install gudhi OR pip install giotto-tda"
            )
    
    if use_gudhi:
        # GUDHI cubical complex
        if filtration_type == 'superlevel':
            # Invert for superlevel
            vol_filt = -volume.copy()
        else:
            vol_filt = volume.copy()
        
        # Create cubical complex
        cc = gd.CubicalComplex(
            dimensions=volume.shape,
            top_dimensional_cells=vol_filt.flatten()
        )
        
        # Compute persistence
        cc.compute_persistence()
        
        diagrams = []
        for dim in dims:
            pairs = cc.persistence_intervals_in_dimension(dim)
            
            if filtration_type == 'superlevel':
                # Undo inversion
                pairs = -pairs[:, [1, 0]]
            
            diagrams.append(pairs)
        
        return diagrams
    
    else:
        # Giotto-TDA
        cubical = CubicalPersistence(
            homology_dimensions=dims,
            n_jobs=1
        )
        
        # Giotto expects (n_samples, D, H, W)
        vol_input = volume[None, ...]
        
        diagrams_list = cubical.fit_transform(vol_input)[0]  # (n_features, 3): (birth, death, dim)
        
        # Split by dimension
        diagrams = []
        for dim in dims:
            mask = diagrams_list[:, 2] == dim
            diag = diagrams_list[mask, :2]  # (n, 2)
            diagrams.append(diag)
        
        return diagrams


def persistence_image(
    diagram: np.ndarray,
    sigma: float = 3.0,
    grid_size: int = 16,
    weight_alpha: float = 1.0,
    birth_range: Optional[Tuple[float, float]] = None,
    pers_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Convert persistence diagram to persistence image.
    
    Uses birth-persistence coordinates: (b, p) where p = d - b
    
    Args:
        diagram: (n, 2) array with [birth, death]
        sigma: Gaussian kernel bandwidth in mm
        grid_size: image resolution (grid_size x grid_size)
        weight_alpha: persistence weighting exponent (higher = emphasize long-lived features)
        birth_range: (min, max) for birth axis; auto-computed if None
        pers_range: (min, max) for persistence axis
    
    Returns:
        persistence image: (grid_size, grid_size) array
    """
    if len(diagram) == 0:
        return np.zeros((grid_size, grid_size))
    
    # Convert to birth-persistence
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    persistences = deaths - births
    
    # Remove points on diagonal (infinite persistence)
    finite_mask = persistences < np.inf
    births = births[finite_mask]
    persistences = persistences[finite_mask]
    
    if len(births) == 0:
        return np.zeros((grid_size, grid_size))
    
    # Define grid ranges
    if birth_range is None:
        b_min, b_max = births.min(), births.max()
        b_margin = (b_max - b_min) * 0.1
        birth_range = (b_min - b_margin, b_max + b_margin)
    
    if pers_range is None:
        p_min, p_max = 0, persistences.max()
        p_margin = p_max * 0.1
        pers_range = (p_min, p_max + p_margin)
    
    # Create grid
    b_edges = np.linspace(birth_range[0], birth_range[1], grid_size + 1)
    p_edges = np.linspace(pers_range[0], pers_range[1], grid_size + 1)
    
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])
    
    B, P = np.meshgrid(b_centers, p_centers, indexing='ij')
    
    # Rasterize with Gaussian kernels
    image = np.zeros((grid_size, grid_size))
    
    for b, p in zip(births, persistences):
        # Weight by persistence^alpha
        weight = p ** weight_alpha
        
        # Gaussian kernel
        gauss = weight * np.exp(-((B - b)**2 + (P - p)**2) / (2 * sigma**2))
        image += gauss
    
    return image


def extract_ph_features(
    mask: np.ndarray,
    rim: np.ndarray,
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    config: Optional[Dict] = None
) -> np.ndarray:
    """
    Extract full persistent homology feature vector for one case.
    
    Computes:
    1. Tumor distance transform → PH dims [0, 1, 2]
    2. Rim intensity → PH dims [0, 1]
    
    Converts each diagram to persistence image, then concatenates and applies PCA.
    
    Args:
        mask: (D, H, W) tumor mask
        rim: (D, H, W) rim mask
        volume: (D, H, W) intensity volume
        spacing: (z, y, x) in mm
        config: topology config with sigma, grid_size, etc.
    
    Returns:
        feature vector: (128,) after PCA
    """
    if config is None:
        config = {
            'sigma_mm': 3.0,
            'grid_size': 16,
            'weight_alpha': 0.75,
            'tumor_dims': [0, 1, 2],
            'rim_dims': [0, 1]
        }
    
    sigma = config.get('sigma_mm', 3.0)
    grid_size = config.get('grid_size', 16)
    weight_alpha = config.get('weight_alpha', 0.75)
    
    # 1. Tumor: distance transform filtration (superlevel)
    dt_tumor = ndi.distance_transform_edt(mask, sampling=spacing)
    
    tumor_diagrams = compute_persistence_diagrams(
        dt_tumor,
        filtration_type='superlevel',
        dims=config.get('tumor_dims', [0, 1, 2]),
        spacing=spacing
    )
    
    # 2. Rim: intensity filtration (sublevel)
    rim_volume = volume * rim
    
    rim_diagrams = compute_persistence_diagrams(
        rim_volume,
        filtration_type='sublevel',
        dims=config.get('rim_dims', [0, 1]),
        spacing=spacing
    )
    
    # Convert all diagrams to persistence images
    all_images = []
    
    for diag in tumor_diagrams:
        pi = persistence_image(diag, sigma=sigma, grid_size=grid_size, weight_alpha=weight_alpha)
        all_images.append(pi.ravel())
    
    for diag in rim_diagrams:
        pi = persistence_image(diag, sigma=sigma, grid_size=grid_size, weight_alpha=weight_alpha)
        all_images.append(pi.ravel())
    
    # Concatenate all PIs
    feature_vec = np.concatenate(all_images)
    
    # Note: In production, apply PCA fitted on training set
    # For now, return raw features (will be 5 diagrams × 256 = 1280 dims)
    # User should fit PCA separately during preprocessing
    
    return feature_vec.astype(np.float32)
