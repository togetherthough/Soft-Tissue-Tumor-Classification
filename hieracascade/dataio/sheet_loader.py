"""Load labels from existing sheet.csv file

This module provides utilities to load dataset indices from the existing sheet.csv
format used in the project. It handles automatic modality detection and file path
resolution for different directory structures.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional


def load_from_sheet_csv(
    sheet_path: str,
    data_root: str,
    label_column: str = 'target',
    study_id_col: str = 'case_id',
    modality_col: Optional[str] = 'modality',
    site_col: Optional[str] = 'site'
) -> List[Dict]:
    """Load dataset index from sheet.csv format
    
    This function reads the sheet.csv file and creates a structured index for training.
    It automatically infers missing information like modality from study IDs when needed.
    
    Args:
        sheet_path: Path to sheet.csv file
        data_root: Root directory containing the data folders
        label_column: Column name for tumor class labels. 
                     Options: 'Diagnosis' (multi-class) or 'Diagnosis_binary' (binary)
                     Default: 'target' for backwards compatibility
        study_id_col: Column name for study ID (default: 'case_id')
        modality_col: Column name for imaging modality CT/MRI (optional)
        site_col: Column name for hospital/site identifier (optional)
        
    Returns:
        List of dicts with keys: study_id, category, modality, path, y_fine, y_coarse, site
    """
    from .utils import FINE_TO_IDX, COARSE_TO_IDX, CLASS_HIERARCHY
    
    # Load the CSV file
    df = pd.read_csv(sheet_path)
    
    # Check if required columns exist
    if study_id_col not in df.columns:
        raise ValueError(f"Column '{study_id_col}' not found in sheet.csv")
    if label_column not in df.columns:
        raise ValueError(
            f"Column '{label_column}' not found in sheet.csv. "
            f"Available columns: {list(df.columns)}. "
            f"Use --label_column to specify one of: 'Diagnosis', 'Diagnosis_binary', etc."
        )
    
    data_root = Path(data_root)
    index = []
    
    # Track unique categories for validation
    unique_categories = set()
    
    for idx, row in df.iterrows():
        study_id = str(row[study_id_col])
        category = str(row[label_column]).lower().strip()
        unique_categories.add(category)
        
        # Get modality from column or infer from study ID
        if modality_col and modality_col in df.columns:
            modality = str(row[modality_col]).upper()
            if modality not in ['CT', 'MRI', 'MR']:
                modality = 'CT'  # Default to CT if invalid
            if modality == 'MR':
                modality = 'MRI'  # Normalize MR to MRI
        else:
            # Try to infer from study_id (eg. if it contains '_MR')
            modality = 'MRI' if '_MR' in study_id else 'CT'
        
        # Get site/hospital identifier
        if site_col and site_col in df.columns:
            site = str(row[site_col])
        else:
            site = category  # Use category as site if not specified
        
        # Try to find the image file in various possible locations
        # Different datasets might have slightly different directory structures
        possible_paths = [
            data_root / category / f"{study_id}_{modality}" / "1" / "NIFTI" / "image.nii.gz",
            data_root / category / study_id / "1" / "NIFTI" / "image.nii.gz",
            data_root / category / f"{study_id}_{modality}" / "NIFTI" / "image.nii.gz",
            data_root / f"{study_id}_{modality}" / "1" / "NIFTI" / "image.nii.gz",
        ]
        
        image_path = None
        for path in possible_paths:
            if path.exists():
                image_path = path
                break
        
        if image_path is None:
            print(f"Warning: Image not found for study {study_id}, skipping")
            continue
        
        # Map category to class indices (both fine and coarse)
        if category not in FINE_TO_IDX:
            print(f"Warning: Unknown category '{category}' for study {study_id}, skipping")
            continue
        
        y_fine = FINE_TO_IDX[category]
        coarse_family = CLASS_HIERARCHY.get(category, 'other')
        y_coarse = COARSE_TO_IDX[coarse_family]
        
        # Add to index
        index.append({
            'study_id': study_id,
            'category': category,
            'modality': modality,
            'path': str(image_path),
            'y_fine': y_fine,
            'y_coarse': y_coarse,
            'site': site,
        })
    
    print(f"\nFound {len(unique_categories)} unique categories: {sorted(unique_categories)}")
    
    return index


def create_index_from_sheet(
    data_root: str = 'data',
    sheet_path: str = 'data/sheet.csv',
    label_column: str = 'target',
    output_csv: Optional[str] = None,
    **kwargs
) -> List[Dict]:
    """Convenience function to create index from sheet.csv
    
    This is a wrapper function that loads the sheet.csv, creates the index,
    and optionally saves it to a processed labels CSV file.
    
    Args:
        data_root: Root data directory
        sheet_path: Path to the sheet.csv file
        label_column: Column to use for labels ('Diagnosis', 'Diagnosis_binary', etc.)
        **kwargs: Additional arguments passed to load_from_sheet_csv
        
    Returns:
        Dataset index (list of dicts)
    """
    from .utils import save_labels_csv, print_dataset_statistics
    
    print(f"Loading labels from {sheet_path}...")
    print(f"Using label column: '{label_column}'")
    
    index = load_from_sheet_csv(sheet_path, data_root, label_column=label_column, **kwargs)
    
    if len(index) == 0:
        raise ValueError("No valid studies found! Check sheet.csv and data paths.")
    
    # Print statistics about the dataset
    print_dataset_statistics(index)
    
    # Optionally save to a processed CSV file
    if output_csv:
        save_labels_csv(index, output_csv)
    
    return index
