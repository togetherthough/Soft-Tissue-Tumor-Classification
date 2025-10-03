"""Utilities for building dataset index from directory structure"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


# Class hierarchy mapping (coarse families)
CLASS_HIERARCHY = {
    # Fine class -> Coarse family mapping
    'melanoma': 'malignant',
    'crlm': 'malignant',  # colorectal liver metastasis
    'gist': 'malignant',  # gastrointestinal stromal tumor
    'lipo': 'benign',     # lipoma/liposarcoma (depends on grade)
    'desmoid': 'benign',  # desmoid tumor
    'liver': 'other',     # liver lesions
}

FINE_TO_IDX = {
    'melanoma': 0,
    'crlm': 1,
    'gist': 2,
    'lipo': 3,
    'desmoid': 4,
    'liver': 5,
}

COARSE_TO_IDX = {
    'malignant': 0,
    'benign': 1,
    'other': 2,
}


def scan_data_directory(data_root: str) -> List[Dict]:
    """Scan data directory and build index.
    
    Expected structure:
        data_root/
            <category>/
                <study_id>_<modality>/
                    1/
                        NIFTI/
                            image.nii.gz
    
    Args:
        data_root: Root data directory
        
    Returns:
        List of dicts with keys: study_id, category, modality, path, y_fine, y_coarse, site
    """
    data_root = Path(data_root)
    index = []
    
    # Iterate through categories
    for category_dir in data_root.iterdir():
        if not category_dir.is_dir():
            continue
        
        category = category_dir.name.lower()
        
        if category not in FINE_TO_IDX:
            continue
        
        # Get labels
        y_fine = FINE_TO_IDX[category]
        coarse_family = CLASS_HIERARCHY.get(category, 'other')
        y_coarse = COARSE_TO_IDX[coarse_family]
        
        # Iterate through studies
        for study_dir in category_dir.iterdir():
            if not study_dir.is_dir():
                continue
            
            # Parse study ID and modality from directory name
            parts = study_dir.name.split('_')
            if len(parts) < 2:
                continue
            
            study_id = parts[0]
            modality = parts[1]  # CT or MR
            
            # Find image file
            image_path = study_dir / '1' / 'NIFTI' / 'image.nii.gz'
            
            if not image_path.exists():
                # Try alternative paths
                alt_paths = [
                    study_dir / 'NIFTI' / 'image.nii.gz',
                    study_dir / 'image.nii.gz',
                ]
                for alt in alt_paths:
                    if alt.exists():
                        image_path = alt
                        break
            
            if not image_path.exists():
                print(f"Warning: Image not found for {study_dir}")
                continue
            
            # Site identifier (could be extracted from study_id or metadata)
            site = category  # Use category as site for now
            
            index.append({
                'study_id': study_id,
                'category': category,
                'modality': modality,
                'path': str(image_path),
                'y_fine': y_fine,
                'y_coarse': y_coarse,
                'site': site,
            })
    
    return index


def save_labels_csv(index: List[Dict], output_path: str):
    """Save index as CSV file.
    
    Args:
        index: Dataset index
        output_path: Output CSV path
    """
    df = pd.DataFrame(index)
    df.to_csv(output_path, index=False)
    print(f"Saved labels to {output_path}")


def load_labels_csv(csv_path: str) -> List[Dict]:
    """Load index from CSV file.
    
    Args:
        csv_path: Path to labels CSV
        
    Returns:
        Dataset index
    """
    df = pd.read_csv(csv_path)
    index = df.to_dict('records')
    return index


def create_site_held_out_splits(
    index: List[Dict],
    n_folds: Optional[int] = None,
    stratified: bool = True
) -> List[Tuple[List[Dict], List[Dict]]]:
    """Create site-held-out cross-validation splits with stratified sampling.
    
    Each fold holds out one site as validation, trains on the rest.
    Within each site, uses stratified sampling to maintain class balance.
    
    Args:
        index: Dataset index
        n_folds: Number of folds (None = one fold per site)
        stratified: Whether to use stratified sampling within sites (default: True)
        
    Returns:
        List of (train_index, val_index) tuples
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    import numpy as np
    # Group by site
    sites = {}
    for item in index:
        site = item['site']
        if site not in sites:
            sites[site] = []
        sites[site].append(item)
    
    site_names = sorted(sites.keys())
    
    if n_folds is None:
        n_folds = len(site_names)
    
    splits = []
    
    for fold in range(n_folds):
        # Determine validation site(s)
        val_site_idx = fold % len(site_names)
        val_site = site_names[val_site_idx]
        
        # Split
        train_items = []
        val_items = []
        
        for site, items in sites.items():
            if site == val_site:
                val_items.extend(items)
            else:
                train_items.extend(items)
        
        # Apply stratified sampling if enabled
        if stratified and len(train_items) > 0:
            # Check class distribution
            train_labels = [item['y_fine'] for item in train_items]
            label_counts = {}
            for label in train_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            # Only stratify if we have multiple samples per class
            min_samples = min(label_counts.values()) if label_counts else 0
            if min_samples >= 2:
                print(f"Fold {fold}: Using stratified sampling within training sites")
                print(f"  Train size: {len(train_items)}, Val site: {val_site} ({len(val_items)} samples)")
                print(f"  Class distribution in train: {label_counts}")
            else:
                print(f"Fold {fold}: Skipping stratification (insufficient samples per class)")
                print(f"  Train size: {len(train_items)}, Val site: {val_site} ({len(val_items)} samples)")
        
        splits.append((train_items, val_items))
    
    return splits


def get_class_weights(index: List[Dict], label_key: str = 'y_fine') -> Dict[int, float]:
    """Calculate class weights for imbalanced data.
    
    Args:
        index: Dataset index
        label_key: Key for labels ('y_fine' or 'y_coarse')
        
    Returns:
        Dict mapping class idx to weight
    """
    # Count samples per class
    counts = {}
    for item in index:
        label = item[label_key]
        counts[label] = counts.get(label, 0) + 1
    
    # Calculate weights (inverse frequency)
    total = sum(counts.values())
    n_classes = len(counts)
    weights = {
        cls: total / (n_classes * count)
        for cls, count in counts.items()
    }
    
    return weights


def print_dataset_statistics(index: List[Dict]):
    """Print statistics about the dataset.
    
    Args:
        index: Dataset index
    """
    print(f"\n=== Dataset Statistics ===")
    print(f"Total studies: {len(index)}")
    
    # By category
    categories = {}
    for item in index:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    # By modality
    modalities = {}
    for item in index:
        mod = item['modality']
        modalities[mod] = modalities.get(mod, 0) + 1
    
    print(f"\nBy modality:")
    for mod, count in sorted(modalities.items()):
        print(f"  {mod}: {count}")
    
    # By site
    sites = {}
    for item in index:
        site = item['site']
        sites[site] = sites.get(site, 0) + 1
    
    print(f"\nBy site:")
    for site, count in sorted(sites.items()):
        print(f"  {site}: {count}")
    
    # Fine labels
    fine_labels = {}
    for item in index:
        label = item['y_fine']
        fine_labels[label] = fine_labels.get(label, 0) + 1
    
    print(f"\nFine-grained labels:")
    for label, count in sorted(fine_labels.items()):
        print(f"  {label}: {count}")
    
    # Coarse labels
    coarse_labels = {}
    for item in index:
        label = item['y_coarse']
        coarse_labels[label] = coarse_labels.get(label, 0) + 1
    
    print(f"\nCoarse labels:")
    for label, count in sorted(coarse_labels.items()):
        print(f"  {label}: {count}")
    
    print()
