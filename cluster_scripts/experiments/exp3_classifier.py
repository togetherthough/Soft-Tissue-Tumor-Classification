#!/usr/bin/env python
"""
Experiment 3: Classification Head on SAM-Med3D Features

This script trains a simple classification head on top of SAM-Med3D encoder
to evaluate whether the pretrained features are discriminative for tumor classification.

Usage:
    python run_experiment3_classification_head.py --config configs/datasets.yaml
    python run_experiment3_classification_head.py --datasets gist lipo --epochs 20
"""

import os
import sys
import site
from pathlib import Path
from typing import Optional, List
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


def run_classification_head_for_dataset(
    dataset_key: str,
    dataset_config: dict,
    repo_root: Path,
    freeze_encoder: bool = True,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    img_size: int = 128,
    num_workers: int = 2,
    output_base: Path = None,
    pooling_strategy: str = 'avg',
    # Lesion filtering parameters
    min_voxels: Optional[int] = None,
    min_dimension: Optional[int] = None,
    min_density: Optional[float] = None,
):
    """
    Run classification head experiment for a single dataset
    
    Args:
        dataset_key: Dataset name (e.g., 'gist', 'lipo')
        dataset_config: Dataset configuration dict from YAML
        repo_root: Repository root path
        freeze_encoder: Whether to freeze SAM-Med3D encoder
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        dropout: Dropout rate
        img_size: Image size
        num_workers: Number of data loader workers
        output_base: Base output directory
    
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
    print(f'DATASET: {dataset_key.upper()}')
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
        print(f"   Download from: https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth")
        print(f"   Save to: {sam3d_root / 'ckpt' / 'sam_med3d_turbo.pth'}")
        checkpoint_path = None
    else:
        print(f"✅ Using checkpoint: {checkpoint_path}")
    
    # Setup output directory with filtering suffix if applicable
    if output_base is None:
        output_base = repo_root / 'results' / 'classification_head'
    
    # Add suffix to output dir if filtering is enabled
    is_filtered = (min_voxels is not None or min_dimension is not None or min_density is not None)
    if is_filtered:
        filter_suffix = "_filtered"
        filter_desc = []
        if min_voxels is not None:
            filter_desc.append(f"v{min_voxels}")
        if min_dimension is not None:
            filter_desc.append(f"d{min_dimension}")
        if min_density is not None:
            filter_desc.append(f"ρ{min_density:.2f}")
        filter_suffix += "_" + "_".join(filter_desc)
        output_dir = output_base / f"{dataset_key}{filter_suffix}"
    else:
        output_dir = output_base / dataset_key
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Output directory: {output_dir}')
    if is_filtered:
        print(f'  Filtered results: voxels>={min_voxels}, dim>={min_dimension}, density>={min_density}')
    
    # Step 1: Prepare dataset
    print(f'\n[1/4] Preparing dataset...')
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
        lesion_filter=lesion_filter,
    )
    
    # Print summary
    print(f'\n{"="*80}')
    print(f'RESULTS FOR {dataset_key.upper()}')
    print(f'{"="*80}')
    print(f'Best Epoch: {results["best_epoch"]}')
    print(f'Best Val AUC: {results["best_auc"]:.4f}')
    metrics = results['final_metrics']
    print(f'Final Accuracy: {metrics.accuracy:.4f}')
    print(f'Final AUC: {metrics.auc:.4f}')
    print(f'{"="*80}\n')
    
    return {
        'dataset': dataset_key,
        'category': category,
        'best_epoch': results['best_epoch'],
        'best_auc': results['best_auc'],
        'final_accuracy': metrics.accuracy,
        'final_auc': metrics.auc,
        'output_dir': str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run Experiment 3: Classification Head on SAM-Med3D'
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
        default='results/classification_head',
        help='Output directory for results (default: results/classification_head)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=None,
        help='Specific datasets to run (default: all in config)'
    )
    parser.add_argument(
        '--freeze-encoder',
        action='store_true',
        default=True,
        help='Freeze SAM-Med3D encoder (default: True)'
    )
    parser.add_argument(
        '--fine-tune',
        action='store_true',
        help='Fine-tune encoder instead of freezing (overrides --freeze-encoder)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size (default: 4)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay (default: 1e-4)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate (default: 0.3)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=128,
        help='Image size (default: 128)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,
        help='Number of data loader workers (default: 2)'
    )
    # Pooling strategy argument
    parser.add_argument(
        '--pooling-strategy',
        type=str,
        choices=['avg', 'multiscale', 'percentile'],
        default='avg',
        help='Pooling strategy for feature aggregation (default: avg)'
    )
    # Lesion filtering arguments
    parser.add_argument(
        '--min-voxels',
        type=int,
        default=None,
        help='Minimum preprocessed voxel count for filtering (default: None - no filtering)'
    )
    parser.add_argument(
        '--min-dimension',
        type=int,
        default=None,
        help='Minimum bounding box dimension for filtering (default: None)'
    )
    parser.add_argument(
        '--min-density',
        type=float,
        default=None,
        help='Minimum lesion density for filtering (default: None)'
    )
    parser.add_argument(
        '--filter-preset',
        type=str,
        choices=['recommended', 'conservative', 'lenient'],
        default=None,
        help='Use preset filtering configuration (overrides individual filters)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    repo_root = _add_repo_root_to_sys_path()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    
    output_base = Path(args.output_dir)
    if not output_base.is_absolute():
        output_base = repo_root / output_base
    
    # Load config
    print(f'\n{"="*80}')
    print(f'EXPERIMENT 3: CLASSIFICATION HEAD ON SAM-MED3D')
    print(f'{"="*80}')
    print(f'Config path: {config_path}')
    print(f'Config exists: {config_path.exists()}')
    print(f'Output base: {output_base}')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    
    assert 'datasets' in cfg and isinstance(cfg['datasets'], dict), \
        'Config must define a datasets: mapping'
    
    all_ds = list(cfg['datasets'].keys())
    datasets_to_run = args.datasets or all_ds
    print(f'Datasets to run: {datasets_to_run}')
    
    # Determine freeze setting
    freeze_encoder = not args.fine_tune if args.fine_tune else args.freeze_encoder
    print(f'Freeze encoder: {freeze_encoder}')
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    
    # Apply filter presets or use individual parameters
    min_voxels = args.min_voxels
    min_dimension = args.min_dimension
    min_density = args.min_density
    
    if args.filter_preset:
        print(f'Using filter preset: {args.filter_preset}')
        if args.filter_preset == 'recommended':
            min_voxels = 500
            min_dimension = 5
            min_density = 0.3
        elif args.filter_preset == 'conservative':
            min_voxels = 1000
            min_dimension = 10
            min_density = 0.3
        elif args.filter_preset == 'lenient':
            min_voxels = 200
            min_dimension = 3
            min_density = None
    
    if min_voxels or min_dimension or min_density:
        print(f'Lesion filtering enabled:')
        print(f'  min_voxels: {min_voxels}')
        print(f'  min_dimension: {min_dimension}')
        print(f'  min_density: {min_density}')
    else:
        print(f'Lesion filtering: DISABLED')
    print()
    
    # Run for each dataset
    results_list = []
    for ds_key in datasets_to_run:
        try:
            ds_cfg = cfg['datasets'][ds_key] or {}
            result = run_classification_head_for_dataset(
                dataset_key=ds_key,
                dataset_config=ds_cfg,
                repo_root=repo_root,
                freeze_encoder=freeze_encoder,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                img_size=args.img_size,
                num_workers=args.num_workers,
                output_base=output_base,
                pooling_strategy=args.pooling_strategy,
                min_voxels=min_voxels,
                min_dimension=min_dimension,
                min_density=min_density,
            )
            results_list.append(result)
            print(f'✅ SUCCESS: {ds_key}')
        except Exception as e:
            print(f'❌ FAILED: {ds_key} - {e}')
            import traceback
            traceback.print_exc()
            results_list.append({
                'dataset': ds_key,
                'error': str(e),
            })
    
    # Save summary
    import pandas as pd
    summary_df = pd.DataFrame(results_list)
    summary_path = output_base / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print(f'\n{"="*80}')
    print(f'EXPERIMENT COMPLETED')
    print(f'{"="*80}')
    print(f'Summary saved to: {summary_path}')
    print('\nResults:')
    print(summary_df.to_string(index=False))
    print(f'{"="*80}\n')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
