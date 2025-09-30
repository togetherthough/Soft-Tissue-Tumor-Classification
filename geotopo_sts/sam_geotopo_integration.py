"""
Integration of SAM-Med3D with GeoTopo-STS for automatic test-time inference.

This allows GeoTopo-STS to work with ONLY CT images at test time by using
SAM-Med3D to automatically generate tumor segmentation masks.
"""

import sys
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import yaml

# Add SAM-Med3D to path
SAM_PATH = Path(__file__).parent.parent / 'SAM-Med3D-main' / 'SAM-Med3D-main'
sys.path.insert(0, str(SAM_PATH))

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


class SAMMed3DSegmenter:
    """
    Wrapper for SAM-Med3D to generate tumor masks for GeoTopo-STS.
    """
    
    def __init__(self, checkpoint_path: str, model_type: str = 'vit_b', device: str = 'cuda'):
        """
        Initialize SAM-Med3D segmenter.
        
        Args:
            checkpoint_path: Path to SAM-Med3D checkpoint (.pth file)
            model_type: Model architecture ('vit_b', 'vit_l', 'vit_h')
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Build SAM model (without checkpoint first)
        self.model = sam_model_registry[model_type]()
        
        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✓ Loaded SAM checkpoint from {checkpoint_path}")
        else:
            print(f"⚠️ No checkpoint provided or file not found, using random weights")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        
        print(f"✓ SAM-Med3D loaded on {self.device}")
    
    def segment_tumor_automatic(self, volume: np.ndarray, 
                                spacing: Tuple[float, float, float],
                                use_everything_mode: bool = True) -> np.ndarray:
        """
        Automatically segment tumor from CT volume.
        
        Args:
            volume: (D, H, W) CT volume
            spacing: (dz, dy, dx) voxel spacing
            use_everything_mode: If True, segment everything (useful for tumor detection)
        
        Returns:
            mask: (D, H, W) binary tumor mask
        """
        D, H, W = volume.shape
        mask = np.zeros_like(volume, dtype=np.uint8)
        
        with torch.no_grad():
            # Process each slice
            for z in range(D):
                slice_2d = volume[z]
                
                # Normalize to 0-255
                slice_norm = self._normalize_slice(slice_2d)
                
                # Convert to 3-channel
                slice_3ch = np.stack([slice_norm] * 3, axis=-1)
                
                # Prepare for SAM
                slice_tensor = self.transform.apply_image(slice_3ch)
                slice_tensor = torch.from_numpy(slice_tensor).permute(2, 0, 1).unsqueeze(0)
                slice_tensor = slice_tensor.to(self.device).float()
                
                # Get embeddings
                features = self.model.image_encoder(slice_tensor)
                
                if use_everything_mode:
                    # Automatic mask generation
                    masks, scores, _ = self.model.mask_decoder.predict_torch(
                        image_embeddings=features,
                        point_coords=None,
                        point_labels=None,
                        multimask_output=True
                    )
                    
                    # Take best mask
                    best_idx = scores.argmax()
                    mask[z] = (masks[0, best_idx].cpu().numpy() > 0.5).astype(np.uint8)
                else:
                    # Need prompt points (will implement if needed)
                    pass
        
        # Post-process: keep largest connected component
        mask = self._keep_largest_component(mask)
        
        return mask
    
    def segment_tumor_with_prompt(self, volume: np.ndarray,
                                  spacing: Tuple[float, float, float],
                                  prompt_point: Tuple[int, int, int],
                                  prompt_label: int = 1) -> np.ndarray:
        """
        Segment tumor using a single point prompt (more accurate).
        
        Args:
            volume: (D, H, W) CT volume
            spacing: voxel spacing
            prompt_point: (z, y, x) coordinate of point inside tumor
            prompt_label: 1 for foreground, 0 for background
        
        Returns:
            mask: (D, H, W) binary tumor mask
        """
        D, H, W = volume.shape
        mask = np.zeros_like(volume, dtype=np.uint8)
        
        z_center, y_center, x_center = prompt_point
        
        with torch.no_grad():
            # Process slices around prompt
            z_range = range(max(0, z_center - 20), min(D, z_center + 20))
            
            for z in z_range:
                slice_2d = volume[z]
                slice_norm = self._normalize_slice(slice_2d)
                slice_3ch = np.stack([slice_norm] * 3, axis=-1)
                
                # Prepare for SAM
                slice_tensor = self.transform.apply_image(slice_3ch)
                slice_tensor = torch.from_numpy(slice_tensor).permute(2, 0, 1).unsqueeze(0)
                slice_tensor = slice_tensor.to(self.device).float()
                
                # Prepare point prompt
                point_coords = torch.tensor([[x_center, y_center]], dtype=torch.float32).to(self.device)
                point_labels = torch.tensor([prompt_label], dtype=torch.int32).to(self.device)
                
                # Get embeddings
                features = self.model.image_encoder(slice_tensor)
                
                # Predict with prompt
                masks, scores, _ = self.model.mask_decoder.predict_torch(
                    image_embeddings=features,
                    point_coords=point_coords.unsqueeze(0),
                    point_labels=point_labels.unsqueeze(0),
                    multimask_output=True
                )
                
                # Take best mask
                best_idx = scores.argmax()
                mask[z] = (masks[0, best_idx].cpu().numpy() > 0.5).astype(np.uint8)
        
        return mask
    
    def _normalize_slice(self, slice_2d: np.ndarray, 
                        window_center: float = 100, 
                        window_width: float = 500) -> np.ndarray:
        """Apply CT windowing and normalize to 0-255."""
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        
        slice_windowed = np.clip(slice_2d, min_val, max_val)
        slice_norm = ((slice_windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        return slice_norm
    
    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component."""
        from scipy import ndimage
        
        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            return mask
        
        # Find largest component
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest_component = sizes.argmax() + 1
        
        return (labeled == largest_component).astype(np.uint8)


class GeoTopoSTS_AutoInference:
    """
    Complete inference pipeline: CT scan → Auto-segmentation → GeoTopo-STS → Prediction
    """
    
    def __init__(self, geotopo_config_path: str, 
                 geotopo_checkpoint_path: Optional[str] = None,
                 sam_checkpoint_path: Optional[str] = None,
                 sam_model_type: str = 'vit_b',
                 device: str = 'cuda',
                 use_sam: bool = True):
        """
        Initialize end-to-end inference pipeline.
        
        Args:
            geotopo_config_path: Path to GeoTopo-STS config.yaml
            geotopo_checkpoint_path: Path to trained GeoTopo-STS model (.pth)
            sam_checkpoint_path: Path to SAM-Med3D checkpoint
            sam_model_type: SAM architecture
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(geotopo_config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize SAM-Med3D
        self.use_sam = use_sam and sam_checkpoint_path is not None
        if self.use_sam:
            self.segmenter = SAMMed3DSegmenter(sam_checkpoint_path, sam_model_type, device)
        else:
            print("⚠️ SAM disabled - will use provided masks (test mode)")
            self.segmenter = None
        
        # Initialize GeoTopo-STS
        from geotopo_sts.models import GeoTopoSTS
        self.model = GeoTopoSTS(self.config['model'])
        
        # Load trained weights if checkpoint provided
        if geotopo_checkpoint_path and Path(geotopo_checkpoint_path).exists():
            checkpoint = torch.load(geotopo_checkpoint_path, map_location=self.device)
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✓ GeoTopo-STS loaded from {geotopo_checkpoint_path}")
        else:
            print("⚠️ No checkpoint provided - using random weights (for testing only!)")
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, ct_volume: np.ndarray,
                spacing: Tuple[float, float, float],
                mask: Optional[np.ndarray] = None,
                prompt_point: Optional[Tuple[int, int, int]] = None,
                return_mask: bool = False) -> Union[int, Tuple[int, np.ndarray]]:
        """
        End-to-end prediction from CT scan only.
        
        Args:
            ct_volume: (D, H, W) CT volume
            spacing: voxel spacing
            prompt_point: Optional point inside tumor for better segmentation
            return_mask: If True, also return the generated mask
        
        Returns:
            predicted_class: Predicted tumor class
            mask: (optional) Generated segmentation mask
        """
        from geotopo_sts.dataio.preprocess import preprocess_case
        from geotopo_sts.geometry import extract_mesh_from_mask, compute_mesh_node_features, build_mesh_graph
        from geotopo_sts.topology import extract_ph_features
        
        # Step 1: Auto-segment tumor with SAM-Med3D (or use provided mask)
        if mask is None:
            if not self.use_sam:
                raise ValueError("No mask provided and SAM is disabled. Either provide a mask or enable SAM.")
            
            print("🔍 Auto-segmenting tumor with SAM-Med3D...")
            if prompt_point is not None:
                mask = self.segmenter.segment_tumor_with_prompt(ct_volume, spacing, prompt_point)
            else:
                mask = self.segmenter.segment_tumor_automatic(ct_volume, spacing)
        else:
            print("✓ Using provided mask")
        
        tumor_volume = mask.sum() * np.prod(spacing) / 1000  # cm³
        print(f"   ✓ Tumor segmented: {tumor_volume:.2f} cm³")
        
        # Step 2: Preprocess with GeoTopo-STS
        print("🔧 Extracting geometry and topology features...")
        preprocessed = preprocess_case(ct_volume, mask, spacing, 'ct', self.config['preprocessing'])
        
        # Extract mesh
        vertices, faces = extract_mesh_from_mask(
            preprocessed['mask'],
            spacing=preprocessed['spacing'],
            target_vertices=self.config['geometry']['mesh']['target_vertices']
        )
        
        if len(vertices) > 0:
            node_features = compute_mesh_node_features(
                vertices, faces,
                preprocessed['volume'],
                preprocessed['mask'],
                preprocessed['rim'],
                spacing=preprocessed['spacing']
            )
            edge_index, _ = build_mesh_graph(vertices, faces)
        else:
            # Fallback for very small tumors
            node_features = np.zeros((100, 9))
            vertices = np.zeros((100, 3))
            edge_index = np.array([[i, (i+1)%100] for i in range(100)]).T
        
        # Extract topology
        try:
            ph_features = extract_ph_features(
                preprocessed['mask'],
                preprocessed['rim'],
                preprocessed['volume'],
                spacing=preprocessed['spacing'],
                config=self.config['topology']
            )
        except:
            ph_features = np.zeros(128)
        
        # Step 3: Prepare batch and predict
        print("🧠 Running GeoTopo-STS inference...")
        batch = {
            'volume': torch.from_numpy(preprocessed['volume'][None, None, ...]).float().to(self.device),
            'mask': torch.from_numpy(preprocessed['mask'][None, None, ...]).float().to(self.device),
            'rim': torch.from_numpy(preprocessed['rim'][None, None, ...]).float().to(self.device),
            'mesh_data': {
                'vertices': torch.from_numpy(vertices).float().to(self.device),
                'features': torch.from_numpy(node_features).float().to(self.device),
                'edge_index': torch.from_numpy(edge_index).long().to(self.device)
            },
            'ph_features': torch.from_numpy(ph_features[None, ...]).float().to(self.device),
        }
        
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=-1)
            pred_class = logits.argmax(dim=-1).item()
            confidence = probs[0, pred_class].item()
        
        print(f"   ✓ Predicted class: {pred_class} (confidence: {confidence:.3f})")
        
        if return_mask:
            return pred_class, mask
        return pred_class
    
    def predict_from_file(self, nifti_path: str, 
                         prompt_point: Optional[Tuple[int, int, int]] = None,
                         mask_path: Optional[str] = None) -> int:
        """
        Predict directly from NIfTI file.
        
        Args:
            nifti_path: Path to CT scan (.nii.gz)
            prompt_point: Optional tumor location hint
            mask_path: Optional path to mask file. If None, will auto-search.
        
        Returns:
            predicted_class: Predicted tumor class
        """
        nifti_path = Path(nifti_path)
        
        # Load CT scan
        nii = nib.load(str(nifti_path))
        volume = nii.get_fdata().astype(np.float32)
        spacing = tuple(nii.header.get_zooms()[:3])
        
        # Try to load mask if SAM is disabled
        mask = None
        if not self.use_sam:
            if mask_path:
                # Use provided mask path
                mask_nii = nib.load(mask_path)
                mask = mask_nii.get_fdata().astype(np.uint8)
                print(f"✓ Loaded mask from {mask_path}")
            else:
                # Auto-search for mask in same directory
                mask_candidates = [
                    nifti_path.parent / 'segmentation.nii.gz',
                    nifti_path.parent / 'mask.nii.gz',
                    nifti_path.parent / 'seg.nii.gz',
                    str(nifti_path).replace('image.nii.gz', 'segmentation.nii.gz'),
                    str(nifti_path).replace('image.nii', 'segmentation.nii'),
                ]
                
                for candidate in mask_candidates:
                    candidate_path = Path(candidate)
                    if candidate_path.exists():
                        mask_nii = nib.load(str(candidate_path))
                        mask = mask_nii.get_fdata().astype(np.uint8)
                        print(f"✓ Auto-loaded mask from {candidate_path.name}")
                        break
                
                if mask is None:
                    raise ValueError(
                        f"SAM is disabled and no mask found for {nifti_path.name}.\n"
                        f"Either enable SAM or provide mask_path parameter."
                    )
        
        return self.predict(volume, spacing, mask=mask, prompt_point=prompt_point)


# Example usage
if __name__ == '__main__':
    """
    Example: Fully automatic inference on GIST CT scans
    """
    
    # Initialize pipeline
    pipeline = GeoTopoSTS_AutoInference(
        geotopo_config_path='config.yaml',
        geotopo_checkpoint_path='./outputs/gist_full/best_model.pth',
        sam_checkpoint_path='../SAM-Med3D-main/SAM-Med3D-main/work_dir/SAM/sam_med3d.pth',
        device='cuda'
    )
    
    # Predict on new case (NO manual mask needed!)
    test_file = r'C:\path\to\new_gist_case\image.nii.gz'
    
    # Option 1: Fully automatic
    prediction = pipeline.predict_from_file(test_file)
    print(f"Predicted class: {prediction}")
    
    # Option 2: With prompt point (more accurate)
    prediction = pipeline.predict_from_file(test_file, prompt_point=(50, 150, 150))
    print(f"Predicted class: {prediction}")
