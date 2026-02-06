#!/usr/bin/env python3
"""
Experiment: Classification Head - Three-Way Preprocessing Comparison

This script runs a comprehensive comparison experiment for classification head:
1. Baseline: Full-volume, no lesion filtering
2. Filtered Baseline: Full-volume with lesion size filtering  
3. ROI-Cropped: Adaptive crop/pad (tumor-centered volumes)

All experiments run on GPU and save results for comparison.
"""

import os
import sys
import site
from pathlib import Path
from typing import Optional
import argparse
import yaml

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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _add_repo_root_to_sys_path():
    """Add repository root to sys.path for imports"""
    here = Path(__file__).parent.parent.resolve()  # Go up from cluster_scripts/
    for base in [here, *here.parents]:
        if (base / 'med3pipe').is_dir():
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            print(f'Added repo root to sys.path: {base}')
            return base
    raise RuntimeError("Could not locate 'med3pipe/' in current or parent directories.")


def run_single_experiment(
    dataset_key: str,
    dataset_config: dict,
    repo_root: Path,
    experiment_name: str,
    use_roi_crop: bool = False,
    roi_margin: Optional[int] = None,
    roi_target_size: Optional[int] = None,
    min_voxels: Optional[int] = None,
    min_dimension: Optional[int] = None,
    min_density: Optional[float] = None,
    pooling_strategy: str = 'percentile',
    freeze_encoder: bool = True,
    num_epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    img_size: int = 128,
    num_workers: int = 2,
    output_base_dir: Path = None,
):
    """
    Run a single classification head experiment with specific preprocessing configuration
    
    Args:
        dataset_key: Dataset name (e.g., 'gist', 'lipo')
        dataset_config: Dataset configuration dict from YAML
        repo_root: Repository root path
        experiment_name: Name of this experiment variant
        use_roi_crop: Whether to use ROI cropping
        roi_margin: ROI margin in voxels (if using ROI crop)
        roi_target_size: Target size after ROI crop
        min_voxels: Minimum voxels for lesion filtering
        min_dimension: Minimum dimension for lesion filtering
        min_density: Minimum density for lesion filtering
        freeze_encoder: Whether to freeze SAM-Med3D encoder
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        dropout: Dropout rate
        img_size: Image size
        num_workers: Number of data loader workers
        output_base_dir: Base output directory
    
    Returns:
        Dictionary with results and metrics
    """
    from med3pipe.data.prepare import (
        prepare_for_sam3d, 
        split_validation, 
        find_default_sam3d_root
    )
    from med3pipe.sam.core import load_labels_from_sheet
    from med3pipe.training.classification_head import run_classification_head_experiment
    from med3pipe.tabular.lesion_filter import LesionSizeFilter
    
    print(f'\n{"="*80}')
    print(f'DATASET: {dataset_key.upper()} - {experiment_name.upper()}')
    print(f'{"="*80}')
    
    # Extract config
    category = dataset_config.get('category', dataset_key)
    ct_name = dataset_config.get('ct_name', f'ct_{dataset_key.upper()}')
    labels_cfg = dataset_config.get('labels', {}) or {}
    split_cfg = dataset_config.get('split', {}) or {}
    extraction_cfg = dataset_config.get('extraction', {}) or {}
    
    # Resolve dataset root
    ds_root_raw = dataset_config.get('dataset_root')
    if ds_root_raw:
        ds_root = Path(ds_root_raw)
        if not ds_root.is_absolute():
            ds_root = (repo_root / ds_root).resolve()
    else:
        # Try common locations
        candidates = [
            repo_root / category,
            repo_root / 'data' / category,
        ]
        ds_root = None
        for cand in candidates:
            if cand.exists():
                ds_root = cand
                break
        if ds_root is None:
            raise FileNotFoundError(
                f"Could not find dataset root for {dataset_key}. "
                f"Tried: {[str(c) for c in candidates]}"
            )
    
    print(f'Dataset root: {ds_root}')
    
    # Get label file
    sheet_csv_rel = labels_cfg.get('sheet_csv', 'sheet.csv')
    if Path(sheet_csv_rel).is_absolute():
        sheet_csv = Path(sheet_csv_rel)
    else:
        # Try relative to dataset root first, then repo root
        candidates = [
            ds_root / sheet_csv_rel,
            repo_root / sheet_csv_rel,
        ]
        sheet_csv = None
        for cand in candidates:
            if cand.exists():
                sheet_csv = cand
                break
        if sheet_csv is None:
            raise FileNotFoundError(
                f"Could not find sheet.csv for {dataset_key}. "
                f"Tried: {[str(c) for c in candidates]}"
            )
    
    print(f'Label file: {sheet_csv}')
    
    # Get other params
    dataset_name = labels_cfg.get('dataset_name')
    subject_col = labels_cfg.get('subject_col', 'Subject')
    label_col = labels_cfg.get('label_col', 'Diagnosis_binary')
    case_suffix = labels_cfg.get('case_suffix', '_CT')
    
    split_ratio = split_cfg.get('ratio', 0.8)
    seed = split_cfg.get('seed', 2025)
    img_size = extraction_cfg.get('img_size', img_size)
    
    # Find SAM-Med3D root and checkpoint
    sam3d_root = find_default_sam3d_root()
    checkpoint_path = sam3d_root / 'ckpt' / 'sam_med3d_turbo.pth'
    if not checkpoint_path.exists():
        checkpoint_path = sam3d_root / 'ckpt' / 'SAM-Med3D-turbo.pth'
    if not checkpoint_path.exists():
        print("⚠️  WARNING: No SAM-Med3D checkpoint found! Model will use random weights.")
        checkpoint_path = None
    else:
        print(f"✅ Using checkpoint: {checkpoint_path}")
    
    # Setup output directory
    if output_base_dir is None:
        output_base_dir = repo_root / 'results' / 'classification_head_comparison'
    
    output_dir = output_base_dir / experiment_name / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Output directory: {output_dir}')
    print(f'Preprocessing config:')
    print(f'  ROI crop: {use_roi_crop}')
    if use_roi_crop:
        print(f'  ROI margin: {roi_margin}')
        print(f'  ROI target size: {roi_target_size}')
    print(f'  Lesion filtering: {min_voxels is not None or min_dimension is not None or min_density is not None}')
    if min_voxels is not None:
        print(f'    min_voxels: {min_voxels}')
    if min_dimension is not None:
        print(f'    min_dimension: {min_dimension}')
    if min_density is not None:
        print(f'    min_density: {min_density}')
    
    # Step 1: Prepare dataset
    print(f'\n[1/4] Preparing dataset...')
    
    # Use different preparation functions based on preprocessing approach
    if use_roi_crop:
        from med3pipe.data.prepare import prepare_for_sam3d_roi_cropped
        # For ROI cropping, use different ct_name to avoid collision
        ct_name_roi = f"{ct_name}_roi"
        prepared, paths = prepare_for_sam3d_roi_cropped(
            dataset_root=ds_root,
            sam3d_root=sam3d_root,
            category=category,
            ct_name=ct_name_roi,
            case_glob=None,
            max_cases=None,
            target_size=roi_target_size or 128,
            margin=roi_margin or 10,
        )
    else:
        prepared, paths = prepare_for_sam3d(
            dataset_root=ds_root,
            sam3d_root=sam3d_root,
            category=category,
            ct_name=ct_name,
            case_glob=None,
            max_cases=None,
        )
    
    print(f'✓ Prepared {prepared} cases')
    
    # Step 2: Create validation split
    print(f'\n[2/4] Creating validation split...')
    split_validation(
        paths,
        split_ratio=split_ratio,
        seed=seed,
        copy=True,
    )
    print(f'✓ Split created (ratio={split_ratio}, seed={seed})')
    
    # Step 3: Load labels
    print(f'\n[3/4] Loading labels...')
    df, lab_map = load_labels_from_sheet(
        sheet_csv=sheet_csv,
        dataset_name=dataset_name,
        subject_col=subject_col,
        label_col=label_col,
        case_suffix=case_suffix,
    )
    print(f'✓ Loaded {len(lab_map)} labels')
    print(f'  Class distribution: {df["label"].value_counts().to_dict()}')
    
    # Step 4: Run classification head experiment
    print(f'\n[4/4] Training classification head...')
    print(f'  Freeze encoder: {freeze_encoder}')
    print(f'  Epochs: {num_epochs}')
    print(f'  Batch size: {batch_size}')
    print(f'  Learning rate: {learning_rate}')
    
    # Create lesion filter if filtering is enabled
    lesion_filter = None
    is_filtered = (min_voxels is not None or min_dimension is not None or min_density is not None)
    if is_filtered:
        lesion_filter = LesionSizeFilter(
            min_voxels=min_voxels,
            min_dimension=min_dimension,
            min_density=min_density,
        )
        print(f'  Lesion filtering: ENABLED')
    
    results = run_classification_head_experiment(
        paths=paths,
        lab_map=lab_map,
        sam3d_root=sam3d_root,
        model_type="vit_b_ori",
        checkpoint=checkpoint_path,
        img_size=img_size,
        device=None,  # Auto-detect
        freeze_encoder=freeze_encoder,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        num_workers=num_workers,
        output_dir=output_dir,
        pooling_strategy=pooling_strategy,
        use_medim=True,  # Use MedIM for model loading (recommended)
        lesion_filter=lesion_filter,
    )
    
    # Print summary
    metrics = results['final_metrics']
    print(f'\n{"="*80}')
    print(f'RESULTS FOR {dataset_key.upper()} - {experiment_name.upper()}')
    print(f'{"="*80}')
    print(f'Best Epoch: {results["best_epoch"]}')
    print(f'Best Val AUC: {results["best_auc"]:.4f}')
    print(f'Final Accuracy: {metrics.accuracy:.4f}')
    print(f'Final AUC: {metrics.auc:.4f}')
    print(f'Final Precision: {metrics.precision:.4f}')
    print(f'Final Recall: {metrics.recall:.4f}')
    print(f'Final F1: {metrics.f1:.4f}')
    print(f'{"="*80}\n')
    
    return {
        'dataset': dataset_key,
        'experiment': experiment_name,
        'best_epoch': results['best_epoch'],
        'best_auc': results['best_auc'],
        'accuracy': metrics.accuracy,
        'auc': metrics.auc,
        'precision': metrics.precision,
        'recall': metrics.recall,
        'f1': metrics.f1,
        'output_dir': str(output_dir),
    }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Classification Head - Three-Way Preprocessing Comparison'
    )
    parser.add_argument(
        '--pooling-strategy',
        type=str,
        choices=['avg', 'multiscale', 'percentile'],
        default='percentile',
        help='Pooling strategy for feature aggregation (default: percentile)'
    )
    args = parser.parse_args()
    
    # Configuration
    repo_root = _add_repo_root_to_sys_path()
    config_path = repo_root / "configs" / "datasets_cluster.yaml"
    if not config_path.exists():
        config_path = repo_root / "configs" / "datasets.yaml"
    
    results_dir = repo_root / "results" / "classification_head_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*80}")
    print(f"THREE-WAY PREPROCESSING COMPARISON - CLASSIFICATION HEAD")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Config: {config_path}")
    print(f"Pooling strategy: {args.pooling_strategy}")
    print(f"{'='*80}\n")
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    
    assert 'datasets' in cfg and isinstance(cfg['datasets'], dict), \
        'Config must define a datasets: mapping'
    
    datasets_to_run = list(cfg['datasets'].keys())
    print(f'Datasets to run: {datasets_to_run}\n')
    
    # Shared experiment parameters
    shared_params = {
        'freeze_encoder': True,
        'num_epochs': 20,
        'batch_size': 4,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.3,
        'img_size': 128,
        'num_workers': 2,
        'output_base_dir': results_dir,
        'pooling_strategy': args.pooling_strategy,
    }
    
    # Lesion filter configuration for filtered experiments
    lesion_filter_params = {
        'min_voxels': 100,
        'min_dimension': 3,
        'min_density': 0.1,
    }
    
    # ROI crop configuration
    roi_crop_params = {
        'roi_margin': 10,
        'roi_target_size': 128,
    }
    
    # Store all results
    all_results = []
    
    # Run experiments for each dataset
    for dataset_key in datasets_to_run:
        dataset_config = cfg['datasets'][dataset_key] or {}
        
        # Experiment 1: Baseline (Full-Volume, No Filtering)
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 1: BASELINE (Full-Volume, No Filtering)")
        print(f"Dataset: {dataset_key}")
        print(f"{'='*80}\n")
        
        try:
            result = run_single_experiment(
                dataset_key=dataset_key,
                dataset_config=dataset_config,
                repo_root=repo_root,
                experiment_name="baseline",
                use_roi_crop=False,
                **shared_params
            )
            all_results.append(result)
            print(f"✅ Baseline completed for {dataset_key}")
        except Exception as e:
            print(f"❌ Baseline failed for {dataset_key}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'dataset': dataset_key,
                'experiment': 'baseline',
                'error': str(e),
            })
        
        # Experiment 2: Filtered Baseline (Full-Volume + Lesion Filtering)
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 2: FILTERED BASELINE (Full-Volume + Lesion Filtering)")
        print(f"Dataset: {dataset_key}")
        print(f"{'='*80}\n")
        
        try:
            result = run_single_experiment(
                dataset_key=dataset_key,
                dataset_config=dataset_config,
                repo_root=repo_root,
                experiment_name="filtered_baseline",
                use_roi_crop=False,
                **lesion_filter_params,
                **shared_params
            )
            all_results.append(result)
            print(f"✅ Filtered baseline completed for {dataset_key}")
        except Exception as e:
            print(f"❌ Filtered baseline failed for {dataset_key}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'dataset': dataset_key,
                'experiment': 'filtered_baseline',
                'error': str(e),
            })
        
        # Experiment 3: ROI-Cropped (Adaptive Crop/Pad)
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 3: ROI-CROPPED (Adaptive Crop/Pad)")
        print(f"Dataset: {dataset_key}")
        print(f"{'='*80}\n")
        
        try:
            result = run_single_experiment(
                dataset_key=dataset_key,
                dataset_config=dataset_config,
                repo_root=repo_root,
                experiment_name="roi_cropped",
                use_roi_crop=True,
                **roi_crop_params,
                **shared_params
            )
            all_results.append(result)
            print(f"✅ ROI-cropped completed for {dataset_key}")
        except Exception as e:
            print(f"❌ ROI-cropped failed for {dataset_key}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'dataset': dataset_key,
                'experiment': 'roi_cropped',
                'error': str(e),
            })
    
    # Generate comparison
    print(f"\n{'='*80}")
    print(f"COMPARISON ANALYSIS")
    print(f"{'='*80}\n")
    
    try:
        import pandas as pd
        
        # Create DataFrame from results
        results_df = pd.DataFrame(all_results)
        
        # Save all results
        all_results_path = results_dir / "all_results.csv"
        results_df.to_csv(all_results_path, index=False)
        print(f"✅ All results saved to: {all_results_path}")
        
        # Create comparison table
        if 'error' not in results_df.columns or results_df['error'].isna().all():
            # Pivot to create comparison table
            comparison = results_df.pivot_table(
                index='dataset',
                columns='experiment',
                values=['accuracy', 'auc', 'f1'],
            )
            
            # Flatten column names
            comparison.columns = [f'{metric}_{exp}' for metric, exp in comparison.columns]
            comparison = comparison.reset_index()
            
            # Calculate improvements vs baseline
            if 'accuracy_baseline' in comparison.columns:
                comparison['accuracy_delta_filtered'] = comparison['accuracy_filtered_baseline'] - comparison['accuracy_baseline']
                comparison['accuracy_delta_roi'] = comparison['accuracy_roi_cropped'] - comparison['accuracy_baseline']
                comparison['auc_delta_filtered'] = comparison['auc_filtered_baseline'] - comparison['auc_baseline']
                comparison['auc_delta_roi'] = comparison['auc_roi_cropped'] - comparison['auc_baseline']
                comparison['f1_delta_filtered'] = comparison['f1_filtered_baseline'] - comparison['f1_baseline']
                comparison['f1_delta_roi'] = comparison['f1_roi_cropped'] - comparison['f1_baseline']
            
            # Save comparison
            comparison_path = results_dir / "preprocessing_comparison.csv"
            comparison.to_csv(comparison_path, index=False)
            
            print("\n" + "="*80)
            print("PREPROCESSING COMPARISON")
            print("="*80)
            print(comparison.to_string(index=False))
            print(f"\n✅ Comparison saved to: {comparison_path}")
            
            # Summary statistics
            if 'accuracy_delta_filtered' in comparison.columns:
                print("\n" + "-"*80)
                print("AVERAGE IMPROVEMENTS vs. BASELINE")
                print("-"*80)
                print("Filtered Baseline:")
                print(f"  Accuracy: {comparison['accuracy_delta_filtered'].mean():+.4f}")
                print(f"  F1 Score: {comparison['f1_delta_filtered'].mean():+.4f}")
                print(f"  ROC AUC:  {comparison['auc_delta_filtered'].mean():+.4f}")
                print("\nROI-Cropped:")
                print(f"  Accuracy: {comparison['accuracy_delta_roi'].mean():+.4f}")
                print(f"  F1 Score: {comparison['f1_delta_roi'].mean():+.4f}")
                print(f"  ROC AUC:  {comparison['auc_delta_roi'].mean():+.4f}")
        
    except Exception as e:
        print(f"⚠️  Could not generate comparison: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {results_dir}")
    print(f"  - all_results.csv: Individual experiment results")
    print(f"  - preprocessing_comparison.csv: Side-by-side comparison")


if __name__ == "__main__":
    main()
