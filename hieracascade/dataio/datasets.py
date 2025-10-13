"""PyTorch Dataset classes for Stage-1 and Stage-2 training"""

import os
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path

from .preprocessing import preprocess_volume, extract_crop, augment_volume_3d
from .proposals import get_crop_centers_from_saliency


class DatasetStage1(Dataset):
    """Dataset for Stage-1 training (full volume → coarse predictions + saliency)."""
    
    def __init__(
        self,
        index: List[Dict],
        data_root: str,
        target_spacing: float = 1.5,
        target_size: Tuple[int, int, int] = (192, 192, 192),
        augment: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            index: List of dicts with keys: study_id, y_fine, y_coarse, site, modality, path
            data_root: Root directory containing data
            target_spacing: Target isotropic spacing in mm
            target_size: Target volume dimensions
            augment: Whether to apply augmentations
            cache_dir: Optional directory to cache preprocessed volumes
        """
        self.index = index
        self.data_root = Path(data_root)
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def __len__(self) -> int:
        return len(self.index)
    
    def _get_cache_path(self, study_id: str) -> Optional[Path]:
        """Get cache file path for preprocessed volume."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{study_id}.npy"
    
    def _load_or_preprocess(self, item: Dict) -> np.ndarray:
        """Load from cache or preprocess volume."""
        cache_path = self._get_cache_path(item['study_id'])
        
        # Try loading from cache
        if cache_path and cache_path.exists():
            return np.load(cache_path)
        
        # Preprocess
        volume = preprocess_volume(
            item['path'],
            item['modality'],
            self.target_spacing,
            self.target_size
        )
        
        # Save to cache
        if cache_path:
            np.save(cache_path, volume)
        
        return volume
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int, str, str]:
        """
        Returns:
            volume: (1, D, H, W) tensor
            y_fine: Fine-grained label
            y_coarse: Coarse family label
            modality_id: 0 for CT, 1 for MRI
            site: Site identifier
            study_id: Study identifier
        """
        item = self.index[idx]
        
        # Load/preprocess volume
        volume = self._load_or_preprocess(item)
        
        # Augment if training
        if self.augment:
            volume = augment_volume_3d(volume)
        
        # Add channel dimension
        volume = volume[np.newaxis, ...]  # (1, D, H, W)
        
        # Convert to tensor
        volume = torch.from_numpy(volume).float()
        
        # Modality ID
        modality_id = 0 if item['modality'].upper() == 'CT' else 1
        
        return (
            volume,
            item['y_fine'],
            item['y_coarse'],
            modality_id,
            item['site'],
            item['study_id']
        )


class DatasetStage2(Dataset):
    """Dataset for Stage-2 training (crops → fine predictions).
    
    Uses Stage-1 model to generate crop proposals on-the-fly or from cache.
    """
    
    def __init__(
        self,
        stage1_dataset: DatasetStage1,
        stage1_model: torch.nn.Module,
        K: int = 8,
        crop_size: int = 96,
        nms_distance: int = 16,
        top_p: float = 0.005,
        cache_centers: bool = True,
        device: str = 'cuda'
    ):
        """
        Args:
            stage1_dataset: DatasetStage1 instance
            stage1_model: Trained Stage-1 model for saliency generation
            K: Number of crops per study
            crop_size: Edge length of crop cubes
            nms_distance: NMS distance threshold
            top_p: Top percentile for peak detection
            cache_centers: Cache crop centers per study
            device: Device for Stage-1 inference
        """
        self.dataset = stage1_dataset
        self.model = stage1_model.eval()
        self.K = K
        self.crop_size = crop_size
        self.nms_distance = nms_distance
        self.top_p = top_p
        self.device = device
        
        self.centers_cache = {} if cache_centers else None
        
        # Move model to device
        self.model = self.model.to(device)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    @torch.no_grad()
    def _get_centers(
        self,
        volume: torch.Tensor,
        study_id: str,
        modality_id: int
    ) -> List[Tuple[int, int, int]]:
        """Get crop centers from saliency map.
        
        Args:
            volume: Volume tensor (1, D, H, W)
            study_id: Study identifier for caching
            modality_id: Modality ID
            
        Returns:
            List of K crop centers (z, y, x)
        """
        # Check cache
        if self.centers_cache is not None and study_id in self.centers_cache:
            return self.centers_cache[study_id]
        
        # Generate saliency
        volume_batch = volume.unsqueeze(0).to(self.device)  # (1, 1, D, H, W)
        m_id = torch.tensor([modality_id], device=self.device)
        
        _, saliency = self.model(volume_batch, m_id)
        saliency = saliency[0, 0]  # (D, H, W)
        
        # Get centers
        centers = get_crop_centers_from_saliency(
            saliency,
            K=self.K,
            nms_distance=self.nms_distance,
            top_p=self.top_p,
            volume_shape=volume.shape[-3:]
        )
        
        # Cache
        if self.centers_cache is not None:
            self.centers_cache[study_id] = centers
        
        return centers
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int, str, str]:
        """
        Returns:
            crops: (K, 1, S, S, S) tensor
            y_fine: Fine-grained label
            y_coarse: Coarse family label
            modality_id: Modality ID
            site: Site identifier
            study_id: Study identifier
        """
        # Get full volume and metadata
        volume, y_fine, y_coarse, modality_id, site, study_id = self.dataset[idx]
        
        # Get crop centers
        centers = self._get_centers(volume, study_id, modality_id)
        
        # Extract crops
        volume_np = volume[0].numpy()  # Remove channel dim for cropping
        crops = []
        
        for center in centers:
            crop = extract_crop(volume_np, center, self.crop_size)
            crops.append(crop)
        
        # Stack crops
        crops = np.stack(crops, axis=0)  # (K, S, S, S)
        crops = crops[:, np.newaxis, ...]  # (K, 1, S, S, S)
        crops = torch.from_numpy(crops).float()
        
        return crops, y_fine, y_coarse, modality_id, site, study_id


class SiteBalancedSampler(Sampler):
    """Sampler that balances batches across sites/hospitals.
    
    Ensures each batch contains examples from different sites to reduce domain bias.
    """
    
    def __init__(
        self,
        index: List[Dict],
        batch_size: int,
        shuffle: bool = True
    ):
        """
        Args:
            index: Dataset index with 'site' field
            batch_size: Batch size
            shuffle: Shuffle within site buckets
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by site
        self.site_buckets = {}
        for i, item in enumerate(index):
            site = item['site']
            if site not in self.site_buckets:
                self.site_buckets[site] = []
            self.site_buckets[site].append(i)
        
        self.sites = list(self.site_buckets.keys())
        self.n_sites = len(self.sites)
        
        # Calculate total batches
        total_samples = sum(len(bucket) for bucket in self.site_buckets.values())
        self.n_batches = (total_samples + batch_size - 1) // batch_size
    
    def __iter__(self):
        # Shuffle within each bucket if needed
        if self.shuffle:
            for site in self.sites:
                np.random.shuffle(self.site_buckets[site])
        
        # Create iterators for each bucket
        bucket_iters = {
            site: iter(self.site_buckets[site])
            for site in self.sites
        }
        
        # Round-robin through sites
        for _ in range(self.n_batches):
            batch = []
            
            while len(batch) < self.batch_size:
                for site in self.sites:
                    if len(batch) >= self.batch_size:
                        break
                    
                    try:
                        idx = next(bucket_iters[site])
                        batch.append(idx)
                    except StopIteration:
                        # Bucket exhausted, restart if shuffling
                        if self.shuffle:
                            np.random.shuffle(self.site_buckets[site])
                        bucket_iters[site] = iter(self.site_buckets[site])
            
            yield from batch
    
    def __len__(self) -> int:
        return self.n_batches * self.batch_size
