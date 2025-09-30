"""
GIST CT Dataset Loader
Loads data from the GIST dataset structure:
  data/gist/gist-XXX_CT/NIFTI/image.nii.gz
  data/gist/gist-XXX_CT/NIFTI/mask.nii.gz (or segmentation.nii.gz)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import json
from tqdm import tqdm


def discover_gist_cases(root_dir: str, pattern: str = "*GIST-*_CT") -> List[Dict]:
    """
    Discover all GIST CT cases in the directory.
    
    Handles multiple structures:
    - GIST-XXX_CT/NIFTI/image.nii.gz
    - GIST-XXX_CT/image.nii.gz (files directly in case folder)
    - gist-XXX_CT/... (lowercase)
    
    Args:
        root_dir: Root directory containing GIST data
        pattern: Pattern to match case directories (default: "*GIST-*_CT")
    
    Returns:
        List of dicts with case information
    """
    root = Path(root_dir)
    cases = []
    
    # Try both uppercase and lowercase patterns
    patterns = [pattern, pattern.lower().replace('gist', 'gist')]
    all_case_dirs = []
    for p in patterns:
        all_case_dirs.extend(root.glob(p))
    
    # Remove duplicates and sort
    all_case_dirs = sorted(set(all_case_dirs))
    
    for case_dir in all_case_dirs:
        case_id = case_dir.name  # e.g., GIST-001_CT or gist-001_CT
        
        # Check for NIFTI subfolder in multiple locations
        # Structure can be: GIST-XXX_CT/1/NIFTI/ or GIST-XXX_CT/NIFTI/ or GIST-XXX_CT/
        search_dir = case_dir
        
        # Check for numbered subfolder (e.g., "1") first
        numbered_dirs = [d for d in case_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if numbered_dirs:
            # Use first numbered directory (usually "1")
            potential_nifti = numbered_dirs[0] / "NIFTI"
            if potential_nifti.exists():
                search_dir = potential_nifti
            else:
                search_dir = numbered_dirs[0]
        else:
            # Check for direct NIFTI subfolder
            nifti_dir = case_dir / "NIFTI"
            if nifti_dir.exists():
                search_dir = nifti_dir
        
        # Look for image file
        image_path = None
        # Try standard names first
        for img_name in ["image.nii.gz", "image.nii", "ct.nii.gz", "ct.nii"]:
            potential_img = search_dir / img_name
            if potential_img.exists():
                image_path = potential_img
                break
        
        # If not found, check for multi-lesion format (image_lesion_0.nii.gz)
        if image_path is None:
            lesion_files = list(search_dir.glob("image_lesion_*.nii.gz"))
            if lesion_files:
                # Use first lesion (lesion_0)
                lesion_0 = [f for f in lesion_files if "lesion_0" in f.name]
                if lesion_0:
                    image_path = lesion_0[0]
                else:
                    image_path = sorted(lesion_files)[0]  # Fallback to first one
        
        # Look for mask file (try multiple names)
        mask_path = None
        for mask_name in ["mask.nii.gz", "segmentation.nii.gz", "mask.nii", "segmentation.nii", "seg.nii.gz", "seg.nii"]:
            potential_mask = search_dir / mask_name
            if potential_mask.exists():
                mask_path = potential_mask
                break
        
        # If not found and this is a multi-lesion case, look for corresponding segmentation
        if mask_path is None and image_path is not None and "lesion" in image_path.name:
            # Extract lesion number from image filename
            lesion_num = image_path.name.split("lesion_")[1].split(".")[0]
            lesion_seg_name = f"segmentation_lesion_{lesion_num}.nii.gz"
            potential_lesion_seg = search_dir / lesion_seg_name
            if potential_lesion_seg.exists():
                mask_path = potential_lesion_seg
        
        if image_path is None:
            print(f"Warning: {case_id} - no image file found in {search_dir}")
            continue
        
        if mask_path is None:
            print(f"Warning: {case_id} - no mask file found in {search_dir}")
            continue
        
        cases.append({
            'case_id': case_id,
            'image_path': str(image_path),
            'mask_path': str(mask_path),
            'search_dir': str(search_dir)
        })
    
    return cases


def load_gist_case(case_info: Dict) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """
    Load a single GIST case.
    
    Args:
        case_info: Dictionary with case information
    
    Returns:
        volume: (D, H, W) CT volume
        mask: (D, H, W) tumor mask
        spacing: (dz, dy, dx) voxel spacing in mm
    """
    # Load image
    image_nii = nib.load(case_info['image_path'])
    volume = image_nii.get_fdata().astype(np.float32)
    
    # Load mask
    mask_nii = nib.load(case_info['mask_path'])
    mask = mask_nii.get_fdata().astype(np.uint8)
    
    # Get spacing
    spacing = tuple(image_nii.header.get_zooms()[:3])
    
    # Ensure mask is binary
    if mask.max() > 1:
        print(f"Warning: {case_info['case_id']} has multi-label mask, converting to binary")
        mask = (mask > 0).astype(np.uint8)
    
    return volume, mask, spacing


def prepare_gist_dataset(
    root_dir: str,
    output_dir: str,
    config_path: str,
    num_workers: int = 4
):
    """
    Prepare GIST dataset for GeoTopo-STS.
    
    Discovers cases, preprocesses them, and saves features.
    
    Args:
        root_dir: Root directory of GIST data
        output_dir: Where to save preprocessed data
        config_path: Path to config.yaml
        num_workers: Number of parallel workers
    """
    from geotopo_sts.preprocess_pipeline import preprocess_single_case
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Discover cases
    print(f"Discovering GIST cases in {root_dir}...")
    cases = discover_gist_cases(root_dir)
    print(f"Found {len(cases)} cases")
    
    if len(cases) == 0:
        print("No cases found! Check your data directory.")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save case list
    with open(output_path / 'case_list.json', 'w') as f:
        json.dump(cases, f, indent=2)
    
    # Preprocess each case
    print("\nPreprocessing cases...")
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for case_info in cases:
            future = executor.submit(
                preprocess_single_case,
                case_info['case_id'],
                Path(case_info['image_path']),
                Path(case_info['mask_path']),
                output_path,
                config,
                'ct'  # modality
            )
            futures.append((case_info['case_id'], future))
        
        # Wait for completion
        success_count = 0
        failed_cases = []
        for case_id, future in tqdm(as_completed([f[1] for f in futures]), total=len(futures)):
            try:
                if future.result():
                    success_count += 1
                else:
                    failed_cases.append(case_id)
            except Exception as e:
                print(f"Error processing {case_id}: {e}")
                failed_cases.append(case_id)
    
    print(f"\nPreprocessing complete: {success_count}/{len(cases)} successful")
    
    if failed_cases:
        print(f"Failed cases: {failed_cases}")
    
    # Create labels file (placeholder - you need to add actual labels)
    print("\nCreating labels file...")
    create_labels_file(cases, output_path)
    
    print(f"\nAll data saved to {output_dir}")


def create_labels_file(cases: List[Dict], output_dir: Path):
    """
    Create labels.txt file for GIST cases.
    
    NOTE: This creates a placeholder. You need to add actual labels!
    """
    labels_file = output_dir / 'labels.txt'
    
    with open(labels_file, 'w') as f:
        f.write("# Format: case_id label\n")
        f.write("# Labels: 0=benign, 1=malignant (example - update with your labels)\n")
        for case in cases:
            # Extract case number from case_id (e.g., gist-001_CT -> 001)
            case_id = case['case_id']
            # Placeholder label - REPLACE WITH ACTUAL LABELS
            label = 0  # Default to 0
            f.write(f"{case_id} {label}\n")
    
    print(f"Created {labels_file}")
    print("WARNING: Labels are placeholders! Update with actual labels from your dataset.")


def create_gist_splits(
    output_dir: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42
):
    """
    Create train/val/test splits for GIST dataset.
    
    Args:
        output_dir: Directory with preprocessed data
        val_split: Validation set proportion
        test_split: Test set proportion
        random_seed: Random seed for reproducibility
    """
    from sklearn.model_selection import train_test_split
    import random
    
    output_path = Path(output_dir)
    
    # Load case list
    with open(output_path / 'case_list.json', 'r') as f:
        cases = json.load(f)
    
    case_ids = [c['case_id'] for c in cases]
    
    # Load labels
    labels = []
    with open(output_path / 'labels.txt', 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            cid, lbl = line.strip().split()
            if cid in case_ids:
                labels.append(int(lbl))
    
    if len(labels) != len(case_ids):
        print("Warning: Number of labels doesn't match number of cases. Using random split.")
        # Random split without stratification
        random.seed(random_seed)
        indices = list(range(len(case_ids)))
        random.shuffle(indices)
        
        n_test = int(len(indices) * test_split)
        n_val = int(len(indices) * val_split)
        
        test_indices = indices[:n_test]
        val_indices = indices[n_test:n_test+n_val]
        train_indices = indices[n_test+n_val:]
        
        train_ids = [case_ids[i] for i in train_indices]
        val_ids = [case_ids[i] for i in val_indices]
        test_ids = [case_ids[i] for i in test_indices]
    else:
        # Stratified split
        train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
            case_ids, labels,
            test_size=test_split,
            stratify=labels,
            random_state=random_seed
        )
        
        val_ratio = val_split / (1 - test_split)
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_ratio,
            stratify=train_val_labels,
            random_state=random_seed
        )
    
    # Save splits
    for name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        with open(output_path / f'{name}_split.txt', 'w') as f:
            f.write('\n'.join(ids))
    
    print(f"\nCreated splits:")
    print(f"  Train: {len(train_ids)} cases")
    print(f"  Val: {len(val_ids)} cases")
    print(f"  Test: {len(test_ids)} cases")


def main():
    """Main function for GIST data preparation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare GIST CT dataset')
    parser.add_argument('--root', type=str, default='../data/gist',
                       help='Root directory of GIST data')
    parser.add_argument('--output', type=str, default='./preprocessed_gist',
                       help='Output directory for preprocessed data')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--create-splits', action='store_true',
                       help='Create train/val/test splits')
    args = parser.parse_args()
    
    # Discover and list cases
    print("=" * 60)
    print("GIST CT Dataset Preparation")
    print("=" * 60)
    
    cases = discover_gist_cases(args.root)
    print(f"\nFound {len(cases)} GIST CT cases:")
    for i, case in enumerate(cases[:5], 1):
        print(f"  {i}. {case['case_id']}")
    if len(cases) > 5:
        print(f"  ... and {len(cases) - 5} more")
    
    # Prepare dataset
    response = input(f"\nProceed with preprocessing? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    prepare_gist_dataset(args.root, args.output, args.config, args.workers)
    
    # Create splits
    if args.create_splits:
        print("\nCreating train/val/test splits...")
        create_gist_splits(args.output)
    
    print("\n" + "=" * 60)
    print("GIST dataset preparation complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Update labels in: {args.output}/labels.txt")
    print(f"2. Train model: python -m geotopo_sts.train --config config.yaml --data {args.output}")


if __name__ == '__main__':
    main()
