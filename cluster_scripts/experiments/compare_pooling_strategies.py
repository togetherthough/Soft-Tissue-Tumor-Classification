#!/usr/bin/env python
"""
Pooling Strategy Comparison Experiment

This script compares different pooling strategies for SAM-Med3D embeddings:
1. Average Pooling (avg): Global average across all spatial dimensions
2. Multiscale Pooling (multiscale): Concatenation at 1x1x1, 2x2x2, 4x4x4 scales
3. Percentile Pooling (percentile): 10th, 25th, 50th, 75th, 90th percentiles

All experiments use ROI-cropped volumes and TabPFN with k-fold cross-validation.

Usage:
    python cluster_scripts/experiments/compare_pooling_strategies.py
    python cluster_scripts/experiments/compare_pooling_strategies.py --datasets gist lipo
    python cluster_scripts/experiments/compare_pooling_strategies.py --roi-margin 30
"""

import os
import sys
import site
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
from datetime import datetime

# Environment setup - reduce thread contention
os.environ['PYTHONNOUSERSITE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Remove user site packages
usr = site.getusersitepackages()
sys.path = [p for p in sys.path if p != usr]

import torch
torch.set_num_threads(1)

import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchio as tio
import SimpleITK as sitk
import yaml
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _add_repo_root_to_sys_path():
    """Add repository root to sys.path for imports"""
    here = Path(__file__).parent.parent.resolve()
    for base in [here, *here.parents]:
        if (base / 'med3pipe').is_dir():
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            print(f'Added repo root to sys.path: {base}')
            return base
    raise RuntimeError("Could not locate 'med3pipe/' in current or parent directories.")


# ============================================================================
# Pooling Strategy Implementations
# ============================================================================

def pool_avg(embedding: torch.Tensor) -> np.ndarray:
    """
    Average Pooling: Global average across all spatial dimensions.
    
    Args:
        embedding: (1, C, D, H, W) tensor
    
    Returns:
        (C,) numpy array
    """
    pooled = embedding.mean(dim=(2, 3, 4))  # (1, C)
    return pooled.squeeze(0).cpu().numpy()


def pool_multiscale(embedding: torch.Tensor) -> np.ndarray:
    """
    Multiscale Pooling: Concatenation of features at multiple spatial resolutions.
    
    Creates features at 3 scales:
    - 1x1x1 (global average)
    - 2x2x2 (8 local regions)
    - 4x4x4 (64 local regions)
    
    Args:
        embedding: (1, C, D, H, W) tensor
    
    Returns:
        (C * 73,) numpy array
    """
    B, C, D, H, W = embedding.shape
    features = []
    
    # Scale 1: 1x1x1 (global average)
    f1 = F.adaptive_avg_pool3d(embedding, (1, 1, 1))
    features.append(f1.view(B, -1))
    
    # Scale 2: 2x2x2 (8 local regions)
    f2 = F.adaptive_avg_pool3d(embedding, (2, 2, 2))
    features.append(f2.view(B, -1))
    
    # Scale 3: 4x4x4 (64 local regions)
    f4 = F.adaptive_avg_pool3d(embedding, (4, 4, 4))
    features.append(f4.view(B, -1))
    
    pooled = torch.cat(features, dim=1)
    return pooled.squeeze(0).cpu().numpy()


def pool_percentile(embedding: torch.Tensor) -> np.ndarray:
    """
    Percentile Pooling: Captures the distribution of feature activations.
    
    Computes multiple percentiles per channel:
    - 10th, 25th, 50th (median), 75th, 90th percentiles
    
    Args:
        embedding: (1, C, D, H, W) tensor
    
    Returns:
        (C * 5,) numpy array
    """
    B, C, D, H, W = embedding.shape
    flat = embedding.view(B, C, -1)
    
    percentiles = [10, 25, 50, 75, 90]
    features = []
    
    for p in percentiles:
        q = p / 100.0
        perc = torch.quantile(flat, q, dim=2)
        features.append(perc)
    
    pooled = torch.cat(features, dim=1)
    return pooled.squeeze(0).cpu().numpy()


POOLING_STRATEGIES = {
    'avg': pool_avg,
    'multiscale': pool_multiscale,
    'percentile': pool_percentile,
}


# ============================================================================
# ROI Cropping Helpers
# ============================================================================

def _znorm_masking_method(x):
    return x > 0


def get_bounding_box(mask_array):
    """Get bounding box of non-zero region in 3D mask."""
    where = np.where(mask_array > 0)
    if len(where[0]) == 0:
        return None
    z_min, z_max = where[0].min(), where[0].max()
    y_min, y_max = where[1].min(), where[1].max()
    x_min, x_max = where[2].min(), where[2].max()
    return (z_min, z_max, y_min, y_max, x_min, x_max)


def add_margin_to_bbox(bbox, margin, shape):
    """Add margin to bounding box, respecting image boundaries."""
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    z_min = max(0, z_min - margin)
    z_max = min(shape[0] - 1, z_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(shape[1] - 1, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(shape[2] - 1, x_max + margin)
    return (z_min, z_max, y_min, y_max, x_min, x_max)


def crop_array_to_roi(array, bbox):
    """Crop array to ROI."""
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    return array[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]


def load_volume_roi_for_sam(img_source, mask_source, img_size=128, roi_margin=30, modality='CT'):
    """Load volume with ROI-centric preprocessing for SAM."""
    from med3pipe.sam.transforms import ResizeLargestTo
    
    if isinstance(img_source, sitk.Image):
        sitk_img = img_source
    else:
        sitk_img = sitk.ReadImage(str(img_source))
    sitk_arr_img, _ = tio.data.io.sitk_to_nib(sitk_img)
    
    if isinstance(mask_source, sitk.Image):
        sitk_mask = mask_source
    else:
        sitk_mask = sitk.ReadImage(str(mask_source))
    sitk_arr_mask, _ = tio.data.io.sitk_to_nib(sitk_mask)
    
    img_np = sitk_arr_img.squeeze()
    mask_np = sitk_arr_mask.squeeze()
    
    bbox = get_bounding_box(mask_np)
    if bbox is None:
        bbox = (0, mask_np.shape[0]-1, 0, mask_np.shape[1]-1, 0, mask_np.shape[2]-1)
    
    bbox = add_margin_to_bbox(bbox, roi_margin, mask_np.shape)
    img_cropped = crop_array_to_roi(img_np, bbox)
    mask_cropped = crop_array_to_roi(mask_np, bbox)
    
    img_tensor = torch.from_numpy(img_cropped).float().unsqueeze(0)
    mask_tensor = torch.from_numpy(mask_cropped).float().unsqueeze(0)
    
    subject_img = tio.Subject(image=tio.ScalarImage(tensor=img_tensor))
    subject_mask = tio.Subject(label=tio.LabelMap(tensor=mask_tensor))
    
    if modality.upper() == 'CT':
        subject_img = tio.Clamp(-1000, 1000)(subject_img)
    
    transform = tio.Compose([
        tio.ToCanonical(),
        ResizeLargestTo(target_size=img_size),
        tio.CropOrPad(target_shape=(img_size, img_size, img_size)),
        tio.ZNormalization(masking_method=_znorm_masking_method),
    ])
    
    mask_transform = tio.Compose([
        tio.ToCanonical(),
        ResizeLargestTo(target_size=img_size),
        tio.CropOrPad(target_shape=(img_size, img_size, img_size)),
    ])
    
    subject_img = transform(subject_img)
    subject_mask = mask_transform(subject_mask)
    
    image = subject_img.image.data.clone().detach().unsqueeze(0).float()
    mask = subject_mask.label.data.squeeze().numpy()
    mask = (mask > 0).astype(float)
    
    return image, mask


# ============================================================================
# Dataset Discovery
# ============================================================================

def infer_case_id(case_dir):
    """Infer case identifier from a NIFTI directory path."""
    case_dir = Path(case_dir).resolve()
    if case_dir.name.lower() == 'nifti':
        parent = case_dir.parent
        grandparent = parent.parent if parent else None
        if grandparent and grandparent.name:
            return grandparent.name
        if parent.name:
            return parent.name
    return case_dir.name


def discover_cases_from_config(config_path, project_root, dataset_filter=None):
    """Discover all dataset cases from config file."""
    from med3pipe.data.prepare import find_case_dirs, find_image_files, find_segmentation_files
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    datasets_cfg = config.get('datasets', {})
    discovered = {}
    
    for dataset_name, cfg in datasets_cfg.items():
        if dataset_filter and dataset_name not in dataset_filter:
            continue
        
        dataset_root = Path(cfg.get('dataset_root', '')).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = (project_root / dataset_root).resolve()
        else:
            dataset_root = dataset_root.resolve()
        
        if not dataset_root.exists():
            print(f"⚠️  Dataset root missing for {dataset_name}: {dataset_root}")
            continue
        
        case_glob = cfg.get('prepare', {}).get('case_glob')
        case_dirs = find_case_dirs(dataset_root, case_glob=case_glob)
        
        cases = []
        for case_dir in case_dirs:
            image_files = find_image_files(case_dir)
            seg_files = find_segmentation_files(case_dir)
            if not image_files or not seg_files:
                continue
            case_id = infer_case_id(case_dir)
            cases.append({
                'case_id': case_id,
                'image_files': image_files,
                'segmentation_files': seg_files,
            })
        
        if not cases:
            print(f"⚠️  No valid cases found for {dataset_name}")
            continue
        
        labels_cfg = cfg.get('labels', {})
        sheet_csv = labels_cfg.get('sheet_csv', 'sheet.csv')
        sheet_path = Path(sheet_csv) if Path(sheet_csv).is_absolute() else dataset_root / sheet_csv
        
        if not sheet_path.exists():
            data_dir = dataset_root.parent
            fallback_path = data_dir / 'sheet.csv'
            if fallback_path.exists():
                sheet_path = fallback_path
        
        discovered[dataset_name] = {
            'dataset': dataset_name,
            'category': cfg.get('category', dataset_name),
            'modality': cfg.get('modality', 'CT'),
            'cases': cases,
            'sheet_csv': sheet_path,
            'dataset_name_in_sheet': labels_cfg.get('dataset_name'),
            'subject_col': labels_cfg.get('subject_col', 'Subject'),
            'label_col': labels_cfg.get('label_col', 'Diagnosis_binary'),
            'case_suffix': labels_cfg.get('case_suffix', '_CT'),
        }
    
    return discovered


# ============================================================================
# Feature Extraction
# ============================================================================

@torch.no_grad()
def extract_embeddings_for_dataset(model, dataset_info, device, img_size=128, roi_margin=30):
    """Extract SAM-Med3D embeddings for all cases in a dataset."""
    from med3pipe.data.prepare import merge_images, merge_segmentations
    
    model.eval()
    embeddings = []
    case_ids = []
    modality = dataset_info.get('modality', 'CT')
    
    for case in tqdm(dataset_info['cases'], desc=f"Extracting {dataset_info['dataset']}"):
        case_id = case['case_id']
        try:
            image_sitk = merge_images(case['image_files'])
            mask_sitk = merge_segmentations(case['segmentation_files'])
            
            image_tensor, _ = load_volume_roi_for_sam(
                image_sitk, mask_sitk,
                img_size=img_size,
                roi_margin=roi_margin,
                modality=modality,
            )
            
            image_tensor = image_tensor.to(device)
            embedding = model.image_encoder(image_tensor)
            
            embeddings.append(embedding.cpu())
            case_ids.append(case_id)
            
        except Exception as e:
            print(f"   ⚠️ Error processing {case_id}: {e}")
            continue
    
    return embeddings, case_ids


def apply_pooling(embeddings, pooling_fn):
    """Apply a pooling function to a list of embeddings."""
    features = []
    for emb in embeddings:
        feat = pooling_fn(emb)
        features.append(feat)
    return np.stack(features)


# ============================================================================
# Labels
# ============================================================================

def load_labels_for_dataset(ds_info, case_ids):
    """Load labels for the given case IDs."""
    from med3pipe.sam.core import load_labels_from_sheet, build_y
    
    sheet_csv = ds_info['sheet_csv']
    if not sheet_csv.exists():
        print(f"   ⚠️ Sheet not found: {sheet_csv}")
        return None, None
    
    case_suffix = ds_info.get('case_suffix', '_CT')
    
    df, lab_map = load_labels_from_sheet(
        sheet_csv,
        dataset_name=ds_info['dataset_name_in_sheet'],
        subject_col=ds_info['subject_col'],
        label_col=ds_info['label_col'],
        case_suffix=case_suffix,
    )
    
    y, missing = build_y(case_ids, lab_map)
    
    if len(y) == 0:
        return None, None
    
    valid_mask = ~np.isnan(y)
    valid_indices = np.where(valid_mask)[0]
    y_valid = y[valid_mask].astype(int)
    
    if len(missing) > 0:
        print(f"   ⚠️ Missing labels for {len(missing)} cases")
    
    return y_valid, valid_indices


# ============================================================================
# TabPFN Pipeline
# ============================================================================

def run_tabpfn_kfold(X, y, n_splits=5, n_components_max=500, random_state=42, device='cpu'):
    """Run TabPFN with k-fold cross-validation."""
    from med3pipe.tabular.tabpfn import ensure_tabpfn_on_sys_path
    ensure_tabpfn_on_sys_path()
    from tabpfn.classifier import TabPFNClassifier
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    all_accs = []
    all_f1s = []
    all_aucs = []
    n_features_pca_final = None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        
        n_components = min(n_components_max, X_train_scaled.shape[1], X_train_scaled.shape[0] - 1)
        if n_components < X_train_scaled.shape[1]:
            pca = PCA(n_components=n_components, random_state=random_state)
            X_train = pca.fit_transform(X_train_scaled)
            X_val = pca.transform(X_val_scaled)
        else:
            X_train = X_train_scaled
            X_val = X_val_scaled
        
        if n_features_pca_final is None:
            n_features_pca_final = X_train.shape[1]
        
        clf = TabPFNClassifier(device=device)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        try:
            proba = clf.predict_proba(X_val)
            if proba.ndim == 1:
                proba = np.stack([1 - proba, proba], axis=-1)
            auc = roc_auc_score(y_val, proba[:, 1])
        except:
            auc = None
        
        all_accs.append(acc)
        all_f1s.append(f1)
        if auc is not None:
            all_aucs.append(auc)
    
    return {
        'accuracy': np.mean(all_accs),
        'accuracy_std': np.std(all_accs),
        'macro_f1': np.mean(all_f1s),
        'macro_f1_std': np.std(all_f1s),
        'roc_auc': np.mean(all_aucs) if all_aucs else None,
        'roc_auc_std': np.std(all_aucs) if all_aucs else None,
        'n_samples': len(y),
        'n_features_original': X.shape[1],
        'n_features_pca': n_features_pca_final,
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_pooling_comparison(
    config_path: Path,
    output_dir: Path,
    dataset_filter: Optional[List[str]] = None,
    pooling_strategies: Optional[List[str]] = None,
    roi_margin: int = 30,
    img_size: int = 128,
    n_splits: int = 5,
    n_components_max: int = 500,
    random_state: int = 42,
):
    """Run the full pooling strategy comparison experiment.
    
    Args:
        config_path: Path to datasets YAML config
        output_dir: Output directory for results
        dataset_filter: Optional list of specific datasets to run
        pooling_strategies: Optional list of pooling strategies to run 
                           (choices: 'avg', 'multiscale', 'percentile').
                           If None, runs all strategies.
        roi_margin: ROI margin in voxels
        img_size: Image size for SAM-Med3D
        n_splits: Number of k-fold CV splits
        n_components_max: Maximum PCA components
        random_state: Random state for reproducibility
    """
    import medim
    from med3pipe.sam.core import find_default_sam3d_root
    
    # Filter pooling strategies if specified
    if pooling_strategies is not None:
        # Validate strategy names
        invalid_strategies = [s for s in pooling_strategies if s not in POOLING_STRATEGIES]
        if invalid_strategies:
            raise ValueError(
                f"Invalid pooling strategies: {invalid_strategies}. "
                f"Valid choices are: {list(POOLING_STRATEGIES.keys())}"
            )
        strategies_to_run = {k: v for k, v in POOLING_STRATEGIES.items() if k in pooling_strategies}
    else:
        strategies_to_run = POOLING_STRATEGIES
    
    print(f'\n{"="*80}')
    print('POOLING STRATEGY COMPARISON EXPERIMENT')
    print(f'{"="*80}')
    print(f'Config: {config_path}')
    print(f'Output: {output_dir}')
    print(f'Pooling strategies: {list(strategies_to_run.keys())}')
    print(f'ROI margin: {roi_margin}')
    print(f'Image size: {img_size}')
    print(f'K-fold splits: {n_splits}')
    
    # Setup
    repo_root = config_path.parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # Load model via medim
    print('\n[1/5] Loading SAM-Med3D model via medim...')
    sam3d_root = find_default_sam3d_root()
    checkpoint_path = sam3d_root / 'ckpt' / 'sam_med3d_turbo.pth'
    
    model = medim.create_model(
        "SAM-Med3D",
        pretrained=True,
        checkpoint_path=str(checkpoint_path)
    ).to(device)
    model.eval()
    print('✅ Model loaded successfully!')
    
    # Discover datasets
    print('\n[2/5] Discovering datasets...')
    datasets = discover_cases_from_config(config_path, repo_root, dataset_filter=dataset_filter)
    print(f'Found {len(datasets)} datasets:')
    for name, info in datasets.items():
        print(f'   - {name}: {len(info["cases"])} cases')
    
    # Extract embeddings
    print('\n[3/5] Extracting embeddings...')
    all_embeddings = {}
    all_case_ids = {}
    
    for ds_name, ds_info in datasets.items():
        print(f'\n--- {ds_name} ---')
        embeddings, case_ids = extract_embeddings_for_dataset(
            model, ds_info, device,
            img_size=img_size, roi_margin=roi_margin
        )
        all_embeddings[ds_name] = embeddings
        all_case_ids[ds_name] = case_ids
        print(f'✅ Extracted {len(embeddings)} embeddings')
    
    # Load labels
    print('\n[4/5] Loading labels...')
    all_labels = {}
    all_valid_indices = {}
    
    for ds_name, ds_info in datasets.items():
        print(f'\n--- {ds_name} ---')
        case_ids = all_case_ids[ds_name]
        y, valid_idx = load_labels_for_dataset(ds_info, case_ids)
        
        if y is not None:
            all_labels[ds_name] = y
            all_valid_indices[ds_name] = valid_idx
            print(f'✅ Loaded {len(y)} labels (classes: {np.bincount(y)})')
    
    # Run experiments
    print('\n[5/5] Running experiments...')
    results = []
    
    for pool_name, pool_fn in strategies_to_run.items():
        print(f'\n{"="*60}')
        print(f'POOLING STRATEGY: {pool_name.upper()}')
        print(f'{"="*60}')
        
        for ds_name in datasets.keys():
            if ds_name not in all_labels:
                print(f'   ⚠️ Skipping {ds_name} (no labels)')
                continue
            
            embeddings = all_embeddings[ds_name]
            if not embeddings:
                continue
            
            # Apply pooling
            X_full = apply_pooling(embeddings, pool_fn)
            valid_idx = all_valid_indices[ds_name]
            y = all_labels[ds_name]
            X = X_full[valid_idx]
            
            print(f'\n   Dataset: {ds_name}')
            print(f'   Features shape: {X.shape}')
            print(f'   Labels: {len(y)} (classes: {np.bincount(y)})')
            
            if len(y) < n_splits * 2:
                print(f'   ⚠️ Too few samples for {n_splits}-fold CV')
                continue
            
            try:
                device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
                metrics = run_tabpfn_kfold(
                    X, y,
                    n_splits=n_splits,
                    n_components_max=n_components_max,
                    random_state=random_state,
                    device=device_str,
                )
                
                print(f'   Results:')
                print(f'      Accuracy: {metrics["accuracy"]:.4f} ± {metrics["accuracy_std"]:.4f}')
                print(f'      Macro F1: {metrics["macro_f1"]:.4f} ± {metrics["macro_f1_std"]:.4f}')
                if metrics['roc_auc'] is not None:
                    print(f'      ROC AUC:  {metrics["roc_auc"]:.4f} ± {metrics["roc_auc_std"]:.4f}')
                
                results.append({
                    'dataset': ds_name,
                    'pooling_strategy': pool_name,
                    'accuracy': metrics['accuracy'],
                    'accuracy_std': metrics['accuracy_std'],
                    'macro_f1': metrics['macro_f1'],
                    'macro_f1_std': metrics['macro_f1_std'],
                    'roc_auc': metrics['roc_auc'],
                    'roc_auc_std': metrics['roc_auc_std'],
                    'n_samples': metrics['n_samples'],
                    'n_features_original': metrics['n_features_original'],
                    'n_features_pca': metrics['n_features_pca'],
                })
                
            except Exception as e:
                print(f'   ❌ Error: {e}')
                import traceback
                traceback.print_exc()
                results.append({
                    'dataset': ds_name,
                    'pooling_strategy': pool_name,
                    'error': str(e),
                })
    
    # Save results
    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_csv = output_dir / f'pooling_comparison_{timestamp}.csv'
    df_results.to_csv(output_csv, index=False)
    print(f'\n✅ Results saved to: {output_csv}')
    
    output_csv_latest = output_dir / 'pooling_comparison_latest.csv'
    df_results.to_csv(output_csv_latest, index=False)
    print(f'✅ Latest results: {output_csv_latest}')
    
    # Print summary
    print(f'\n{"="*80}')
    print('RESULTS SUMMARY')
    print(f'{"="*80}')
    print(df_results.to_string(index=False))
    
    # Pivot tables
    if 'accuracy' in df_results.columns and df_results['accuracy'].notna().any():
        print(f'\n{"="*60}')
        print('ACCURACY BY POOLING STRATEGY')
        print(f'{"="*60}')
        pivot_acc = df_results.pivot(
            index='dataset',
            columns='pooling_strategy',
            values='accuracy'
        )
        print(pivot_acc.to_string())
        
        print(f'\n{"="*60}')
        print('ROC AUC BY POOLING STRATEGY')
        print(f'{"="*60}')
        pivot_auc = df_results.pivot(
            index='dataset',
            columns='pooling_strategy',
            values='roc_auc'
        )
        print(pivot_auc.to_string())
        
        # Average across datasets
        print(f'\n{"="*60}')
        print('AVERAGE PERFORMANCE BY POOLING STRATEGY')
        print(f'{"="*60}')
        for pool_name in strategies_to_run.keys():
            pool_data = df_results[df_results['pooling_strategy'] == pool_name]
            if pool_data['accuracy'].notna().any():
                avg_acc = pool_data['accuracy'].mean()
                avg_auc = pool_data['roc_auc'].mean() if pool_data['roc_auc'].notna().any() else None
                print(f'{pool_name}: Accuracy={avg_acc:.4f}', end='')
                if avg_auc is not None:
                    print(f', ROC AUC={avg_auc:.4f}')
                else:
                    print()
    
    return df_results


def main():
    parser = argparse.ArgumentParser(
        description='Run Pooling Strategy Comparison Experiment'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/datasets.yaml',
        help='Path to datasets config YAML (default: configs/datasets.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/pooling_comparison',
        help='Output directory for results (default: results/pooling_comparison)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=None,
        help='Specific datasets to run (default: all in config)'
    )
    parser.add_argument(
        '--pooling-strategies',
        type=str,
        nargs='+',
        choices=['avg', 'multiscale', 'percentile'],
        default=None,
        help='Specific pooling strategies to run (choices: avg, multiscale, percentile). '
             'If not specified, runs all strategies.'
    )
    parser.add_argument(
        '--roi-margin',
        type=int,
        default=30,
        help='ROI margin in voxels (default: 30)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=128,
        help='Image size for SAM-Med3D (default: 128)'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Number of k-fold cross-validation splits (default: 5)'
    )
    parser.add_argument(
        '--n-components-max',
        type=int,
        default=500,
        help='Maximum PCA components (default: 500)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    repo_root = _add_repo_root_to_sys_path()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    
    # Run experiment
    run_pooling_comparison(
        config_path=config_path,
        output_dir=output_dir,
        dataset_filter=args.datasets,
        pooling_strategies=args.pooling_strategies,
        roi_margin=args.roi_margin,
        img_size=args.img_size,
        n_splits=args.n_splits,
        n_components_max=args.n_components_max,
        random_state=args.random_state,
    )
    
    print(f'\n{"="*80}')
    print('EXPERIMENT COMPLETED SUCCESSFULLY')
    print(f'{"="*80}\n')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
