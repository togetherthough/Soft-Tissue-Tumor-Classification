"""
Complete preprocessing pipeline for GeoTopo-STS
Converts raw medical images to preprocessed features for training
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from .dataio.preprocess import preprocess_case
from .geometry import (
    extract_mesh_from_mask,
    compute_mesh_node_features,
    build_mesh_graph
)
from .topology import extract_ph_features


def preprocess_single_case(
    case_id: str,
    volume_path: Path,
    mask_path: Path,
    output_dir: Path,
    config: Dict,
    modality: str = 'mri'
) -> bool:
    """
    Preprocess a single case and save all features.
    
    Args:
        case_id: unique case identifier
        volume_path: path to raw volume (nifti/npy)
        mask_path: path to tumor mask
        output_dir: where to save preprocessed data
        config: preprocessing configuration
        modality: 'mri' or 'ct'
    
    Returns:
        True if successful
    """
    try:
        # Load data
        if str(volume_path).endswith('.npy'):
            volume = np.load(volume_path)
            mask = np.load(mask_path)
            spacing = tuple(config.get('original_spacing', [1.5, 1.5, 1.5]))
        else:
            # Load NIfTI
            import nibabel as nib
            vol_nii = nib.load(volume_path)
            mask_nii = nib.load(mask_path)
            
            volume = vol_nii.get_fdata()
            mask = mask_nii.get_fdata().astype(np.uint8)
            spacing = tuple(vol_nii.header.get_zooms())
        
        # 1. Preprocess volume (resample, normalize, crop, rim)
        preprocessed = preprocess_case(
            volume, mask, spacing,
            modality=modality,
            config=config.get('preprocessing', {})
        )
        
        # Save preprocessed voxel data
        (output_dir / 'volumes').mkdir(exist_ok=True, parents=True)
        (output_dir / 'masks').mkdir(exist_ok=True, parents=True)
        (output_dir / 'rims').mkdir(exist_ok=True, parents=True)
        
        np.save(output_dir / 'volumes' / f'{case_id}.npy', preprocessed['volume'])
        np.save(output_dir / 'masks' / f'{case_id}.npy', preprocessed['mask'])
        np.save(output_dir / 'rims' / f'{case_id}.npy', preprocessed['rim'])
        
        # 2. Extract mesh geometry
        if config.get('geometry', {}).get('mesh', {}).get('enabled', True):
            mesh_config = config['geometry']['mesh']
            
            vertices, faces = extract_mesh_from_mask(
                preprocessed['mask'],
                spacing=preprocessed['spacing'],
                smooth_kernel=tuple(mesh_config.get('smooth_kernel', [3, 3, 3])),
                laplacian_iters=mesh_config.get('laplacian_smooth_iters', 10),
                laplacian_lambda=mesh_config.get('laplacian_lambda', 0.5),
                target_vertices=mesh_config.get('target_vertices', 8000)
            )
            
            if len(vertices) > 0:
                # Compute node features
                node_features = compute_mesh_node_features(
                    vertices, faces,
                    preprocessed['volume'],
                    preprocessed['mask'],
                    preprocessed['rim'],
                    spacing=preprocessed['spacing']
                )
                
                # Build graph
                edge_index, edge_attr = build_mesh_graph(
                    vertices, faces,
                    k_neighbors=mesh_config.get('edge_knn', 8)
                )
                
                # Save mesh data
                (output_dir / 'meshes').mkdir(exist_ok=True, parents=True)
                np.savez(
                    output_dir / 'meshes' / f'{case_id}.npz',
                    vertices=vertices,
                    faces=faces,
                    features=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr
                )
        
        # 3. Extract topology features
        if config.get('topology', {}).get('enabled', True):
            topo_config = config.get('topology', {})
            
            ph_features = extract_ph_features(
                preprocessed['mask'],
                preprocessed['rim'],
                preprocessed['volume'],
                spacing=preprocessed['spacing'],
                config=topo_config
            )
            
            # Save topology features
            (output_dir / 'topology').mkdir(exist_ok=True, parents=True)
            np.save(output_dir / 'topology' / f'{case_id}.npy', ph_features)
        
        return True
    
    except Exception as e:
        print(f"Error processing {case_id}: {e}")
        return False


def create_splits(
    case_ids: List[str],
    labels: List[int],
    output_dir: Path,
    method: str = 'random',
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42
):
    """
    Create train/val/test splits and save.
    
    Args:
        case_ids: list of case IDs
        labels: corresponding labels
        output_dir: where to save split files
        method: 'random' or 'stratified'
        val_size: validation set proportion
        test_size: test set proportion
        random_seed: random seed
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
        case_ids, labels,
        test_size=test_size,
        stratify=labels if method == 'stratified' else None,
        random_state=random_seed
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_ratio,
        stratify=train_val_labels if method == 'stratified' else None,
        random_state=random_seed
    )
    
    # Save splits
    for name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        with open(output_dir / f'{name}_split.txt', 'w') as f:
            f.write('\n'.join(ids))
    
    print(f"Created splits: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess data for GeoTopo-STS')
    parser.add_argument('--config', type=str, required=True, help='Config YAML')
    parser.add_argument('--input', type=str, required=True, help='Input data directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--modality', type=str, default='mri', choices=['mri', 'ct'])
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--create-splits', action='store_true', help='Create train/val/test splits')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all cases
    # Assumes structure: input_dir/volumes/*.nii.gz and input_dir/masks/*.nii.gz
    volume_files = sorted((input_dir / 'volumes').glob('*.nii.gz')) or \
                   sorted((input_dir / 'volumes').glob('*.npy'))
    
    print(f"Found {len(volume_files)} cases to process")
    
    # Process in parallel
    futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for vol_path in volume_files:
            case_id = vol_path.stem.replace('.nii', '')
            
            # Find corresponding mask
            mask_path = input_dir / 'masks' / vol_path.name
            if not mask_path.exists():
                mask_path = input_dir / 'masks' / f'{case_id}.npy'
            
            if not mask_path.exists():
                print(f"Warning: No mask found for {case_id}, skipping")
                continue
            
            future = executor.submit(
                preprocess_single_case,
                case_id, vol_path, mask_path,
                output_dir, config, args.modality
            )
            futures.append((case_id, future))
        
        # Wait for completion
        success_count = 0
        for case_id, future in tqdm(as_completed([f[1] for f in futures]), total=len(futures)):
            if future.result():
                success_count += 1
    
    print(f"\nPreprocessing complete: {success_count}/{len(futures)} successful")
    
    # Create splits if requested
    if args.create_splits:
        print("\nCreating train/val/test splits...")
        
        # Load labels (assumes labels.txt with format: case_id label)
        label_file = input_dir / 'labels.txt'
        if label_file.exists():
            case_ids = []
            labels = []
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        cid, lbl = line.strip().split()
                        case_ids.append(cid)
                        labels.append(int(lbl))
            
            create_splits(
                case_ids, labels, output_dir,
                method=config.get('evaluation', {}).get('split', {}).get('method', 'stratified'),
                val_size=config.get('evaluation', {}).get('split', {}).get('val_size', 0.15),
                test_size=config.get('evaluation', {}).get('split', {}).get('test_size', 0.15),
                random_seed=config.get('evaluation', {}).get('split', {}).get('random_seed', 42)
            )
        else:
            print(f"Warning: {label_file} not found, skipping split creation")
    
    print(f"\nAll data saved to {output_dir}")


if __name__ == '__main__':
    main()
