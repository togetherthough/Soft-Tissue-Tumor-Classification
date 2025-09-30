"""
PyTorch Dataset for GeoTopo-STS
Handles loading and augmentation of 3D medical images with geometry/topology features
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Callable, Tuple, Any
import os
from pathlib import Path


class GeoTopoDataset(Dataset):
    """
    Dataset for GeoTopo-STS pipeline.
    
    Expected to load preprocessed data including:
    - 3D volume crops
    - Tumor masks
    - Peritumoral rim masks
    - Mesh features (vertices, faces, node features)
    - PH features (persistence images)
    - Labels and metadata
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        load_mesh: bool = True,
        load_topology: bool = True,
        load_skeleton: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            data_root: path to preprocessed data directory
            split: 'train', 'val', or 'test'
            transform: augmentation transform
            load_mesh: whether to load mesh geometry
            load_topology: whether to load PH features
            load_skeleton: whether to load skeleton graph
            config: dataset configuration
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.load_mesh = load_mesh
        self.load_topology = load_topology
        self.load_skeleton = load_skeleton
        self.config = config or {}
        
        # Load split file
        split_file = self.data_root / f"{split}_split.txt"
        if split_file.exists():
            with open(split_file, 'r') as f:
                self.case_ids = [line.strip() for line in f if line.strip()]
        else:
            # Fallback: find all cases
            self.case_ids = self._discover_cases()
        
        print(f"Loaded {len(self.case_ids)} cases for split '{split}'")
    
    def _discover_cases(self) -> List[str]:
        """Discover case IDs from data directory"""
        volume_dir = self.data_root / 'volumes'
        if volume_dir.exists():
            return [f.stem for f in volume_dir.glob('*.npy')]
        return []
    
    def __len__(self) -> int:
        return len(self.case_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load one case with all modalities.
        
        Returns:
            dict with keys:
                - 'volume': (C, D, H, W) tensor
                - 'mask': (1, D, H, W) tensor
                - 'rim': (1, D, H, W) tensor
                - 'mesh_data': dict with vertices, features, edges (if load_mesh)
                - 'ph_features': (F,) tensor (if load_topology)
                - 'label': int
                - 'case_id': str
                - 'metadata': dict
        """
        case_id = self.case_ids[idx]
        
        # Load volume data
        volume = self._load_npy(f'volumes/{case_id}.npy')
        mask = self._load_npy(f'masks/{case_id}.npy')
        rim = self._load_npy(f'rims/{case_id}.npy')
        
        # Load label
        label_file = self.data_root / 'labels' / f'{case_id}.txt'
        if label_file.exists():
            label = int(label_file.read_text().strip())
        else:
            label = 0  # default
        
        # Prepare output dict
        sample = {
            'volume': torch.from_numpy(volume).float(),
            'mask': torch.from_numpy(mask).float(),
            'rim': torch.from_numpy(rim).float(),
            'label': label,
            'case_id': case_id
        }
        
        # Ensure correct shapes (C, D, H, W)
        if sample['volume'].ndim == 3:
            sample['volume'] = sample['volume'].unsqueeze(0)  # add channel dim
        if sample['mask'].ndim == 3:
            sample['mask'] = sample['mask'].unsqueeze(0)
        if sample['rim'].ndim == 3:
            sample['rim'] = sample['rim'].unsqueeze(0)
        
        # Load mesh features
        if self.load_mesh:
            mesh_file = self.data_root / 'meshes' / f'{case_id}.npz'
            if mesh_file.exists():
                mesh_data = np.load(mesh_file, allow_pickle=True)
                sample['mesh_data'] = {
                    'vertices': torch.from_numpy(mesh_data['vertices']).float(),  # (N, 3)
                    'features': torch.from_numpy(mesh_data['features']).float(),  # (N, F)
                    'edge_index': torch.from_numpy(mesh_data['edge_index']).long()  # (2, E)
                }
            else:
                # Placeholder if mesh not available
                sample['mesh_data'] = None
        
        # Load topology features
        if self.load_topology:
            topo_file = self.data_root / 'topology' / f'{case_id}.npy'
            if topo_file.exists():
                ph_features = np.load(topo_file)
                sample['ph_features'] = torch.from_numpy(ph_features).float()
            else:
                sample['ph_features'] = torch.zeros(128)  # placeholder
        
        # Load skeleton (if enabled)
        if self.load_skeleton:
            skel_file = self.data_root / 'skeletons' / f'{case_id}.npz'
            if skel_file.exists():
                skel_data = np.load(skel_file, allow_pickle=True)
                sample['skeleton_data'] = {
                    'nodes': torch.from_numpy(skel_data['nodes']).float(),
                    'features': torch.from_numpy(skel_data['features']).float(),
                    'edge_index': torch.from_numpy(skel_data['edge_index']).long()
                }
            else:
                sample['skeleton_data'] = None
        
        # Apply augmentation
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def _load_npy(self, rel_path: str) -> np.ndarray:
        """Load .npy file"""
        full_path = self.data_root / rel_path
        if full_path.exists():
            return np.load(full_path)
        else:
            raise FileNotFoundError(f"File not found: {full_path}")


class Compose:
    """Compose multiple transforms"""
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomRotation3D:
    """Random 3D rotation augmentation"""
    def __init__(self, degrees: float = 25.0, prob: float = 0.5):
        self.degrees = degrees
        self.prob = prob
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.rand() > self.prob:
            return sample
        
        # Random rotation angles
        angles = np.random.uniform(-self.degrees, self.degrees, size=3)
        
        # Apply to volume, mask, rim
        from scipy.ndimage import rotate
        for key in ['volume', 'mask', 'rim']:
            if key in sample:
                vol = sample[key].numpy() if isinstance(sample[key], torch.Tensor) else sample[key]
                # Rotate around each axis
                for axis, angle in enumerate(angles):
                    axes = [(i, j) for i, j in [(1, 2), (2, 3), (1, 3)]][axis]
                    vol = rotate(vol, angle, axes=axes, reshape=False, order=1, mode='constant')
                sample[key] = torch.from_numpy(vol).float()
        
        # TODO: also rotate mesh vertices if present
        
        return sample


class RandomFlip3D:
    """Random 3D flipping"""
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for axis in range(3):  # D, H, W
            if np.random.rand() < self.prob:
                for key in ['volume', 'mask', 'rim']:
                    if key in sample:
                        sample[key] = torch.flip(sample[key], dims=[axis + 1])  # +1 for channel dim
        
        return sample


class IntensityAugmentation:
    """Intensity-based augmentation (gamma, noise)"""
    def __init__(
        self,
        gamma_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.02,
        prob: float = 0.5
    ):
        self.gamma_range = gamma_range
        self.noise_std = noise_std
        self.prob = prob
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.rand() > self.prob:
            return sample
        
        volume = sample['volume']
        
        # Gamma augmentation
        if np.random.rand() < 0.5:
            gamma = np.random.uniform(*self.gamma_range)
            volume = torch.sign(volume) * torch.pow(torch.abs(volume) + 1e-7, gamma)
        
        # Gaussian noise
        if np.random.rand() < 0.5:
            noise = torch.randn_like(volume) * self.noise_std
            volume = volume + noise
        
        sample['volume'] = volume
        return sample


def get_train_transforms(config: Optional[Dict] = None):
    """Get training augmentation transforms"""
    if config is None:
        config = {
            'rotation_degrees': 25,
            'flip_prob': 0.5,
            'gamma_range': [0.8, 1.2],
            'noise_std': 0.02
        }
    
    return Compose([
        RandomRotation3D(degrees=config.get('rotation_degrees', 25), prob=0.5),
        RandomFlip3D(prob=config.get('flip_prob', 0.5)),
        IntensityAugmentation(
            gamma_range=tuple(config.get('gamma_range', [0.8, 1.2])),
            noise_std=config.get('noise_std', 0.02),
            prob=0.5
        )
    ])


def get_val_transforms():
    """Get validation transforms (no augmentation)"""
    return None
