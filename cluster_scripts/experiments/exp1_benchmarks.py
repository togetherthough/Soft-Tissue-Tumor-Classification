#!/usr/bin/env python
"""
Experiment 1: Head-to-head Benchmarks (Med3-TabPFN / Med3-LoCalPFN vs 3D baselines)

This script runs the benchmarks from notebooks/Experiment1-Benchmarks.ipynb
as a standalone script suitable for batch execution on HPC/GPU clusters.

Methods compared:
- Med3-TabPFN (run_multi_tabpfn)
- Med3-LoCalPFN (run_multi_localpfn)
- DenseNet121-3D (train_eval_densenet121_3d)
- ViT-3D (train_eval_vit_3d)
"""

import os
import sys
import site
from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse

# Environment setup - reduce thread contention
os.environ['PYTHONNOUSERSITE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Remove user site packages
usr = site.getusersitepackages()
sys.path = [p for p in sys.path if p != usr]

print("[DEBUG] Importing torch...", flush=True)
import torch
torch.set_num_threads(1)
print("[DEBUG] Torch imported successfully", flush=True)

print("[DEBUG] Importing yaml and pandas...", flush=True)
import yaml
import pandas as pd
print("[DEBUG] YAML and pandas imported successfully", flush=True)


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


def _resolve_dataset_root(dataset_root: Optional[str], category: str, project_root: Path) -> Path:
    """Resolve dataset root path"""
    if dataset_root is not None:
        p = Path(dataset_root)
        if p.is_absolute() and p.exists():
            return p
        cand = (project_root / p).resolve()
        if cand.exists():
            return cand
    c1 = (project_root / category).resolve()
    if c1.exists():
        return c1
    c2 = (project_root / 'data' / category).resolve()
    if c2.exists():
        return c2
    raise FileNotFoundError(f'Could not resolve dataset_root for {category!r}. '
                          f'Tried: {dataset_root!r}, {c1}, {c2}')


def run_experiment(
    config_path: Path,
    outputs_base: Path,
    dataset_filter: Optional[List[str]] = None,
    skip_tabpfn: bool = False,
    skip_localpfn: bool = False,
    skip_baselines: bool = False,
    epochs_3d: int = 4,
    dry_run: bool = False,
    n_splits: int = 5,
    # Pooling
    pooling_strategy: str = 'avg',
    # ROI cropping parameters
    use_roi_crop: bool = False,
    roi_margin: int = 10,
    roi_target_size: int = 128,
    # MedIM integration
    use_medim: bool = True,
    # Lesion filtering parameters
    min_voxels: Optional[int] = None,
    min_dimension: Optional[int] = None,
    min_density: Optional[float] = None,
):
    """
    Run the full benchmark experiment
    
    Args:
        config_path: Path to datasets.yaml config
        pooling_strategy: Pooling strategy to use ('avg', 'multiscale', or 'percentile')
        outputs_base: Base directory for outputs
        dataset_filter: Optional list of dataset names to run (None = all)
        skip_tabpfn: Skip TabPFN method
        skip_localpfn: Skip LoCalPFN method
        skip_baselines: Skip 3D baseline methods
        epochs_3d: Number of epochs for 3D models training
        dry_run: Print what will run without executing
        n_splits: Number of k-fold cross-validation splits (default: 5)
        use_roi_crop: Whether to use ROI cropping (tumor-centered volumes)
        roi_margin: Margin around tumor for ROI cropping (in voxels)
        roi_target_size: Target size after ROI cropping
        use_medim: Whether to use medim for model loading (recommended)
        min_voxels: Minimum voxel count for filtering
        min_dimension: Minimum dimension for filtering
        min_density: Minimum density for filtering
    """
    
    # Import after path setup
    print("[DEBUG] Importing med3pipe modules...", flush=True)
    from med3pipe.pipelines import run_multi_tabpfn, run_multi_localpfn
    print("[DEBUG] Imported pipelines", flush=True)
    from med3pipe.data.prepare import Sam3DPaths, find_default_sam3d_root
    print("[DEBUG] Imported data.prepare", flush=True)
    from med3pipe.vision.v3d import train_eval_densenet121_3d, train_eval_vit_3d
    from med3pipe.tabular.lesion_filter import LesionSizeFilter
    print("[DEBUG] All med3pipe imports complete", flush=True)
    
    print(f'\n{"="*80}')
    print(f'EXPERIMENT 1: BENCHMARKS')
    print(f'{"="*80}')
    print(f'Config path: {config_path}')
    print(f'Config exists: {config_path.exists()}')
    print(f'Outputs base: {outputs_base}')
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    
    assert 'datasets' in cfg and isinstance(cfg['datasets'], dict), \
        'Config must define a datasets: mapping'
    
    all_ds = list(cfg['datasets'].keys())
    datasets_to_run = dataset_filter or all_ds
    print(f'Datasets to run: {datasets_to_run}')
    
    # Create lesion filter if filtering is enabled
    lesion_filter = None
    is_filtered = (min_voxels is not None or min_dimension is not None or min_density is not None)
    if is_filtered:
        lesion_filter = LesionSizeFilter(
            min_voxels=min_voxels,
            min_dimension=min_dimension,
            min_density=min_density,
        )
        print(f'\nLesion filtering enabled:')
        print(f'  min_voxels: {min_voxels}')
        print(f'  min_dimension: {min_dimension}')
        print(f'  min_density: {min_density}')
    else:
        print(f'\nLesion filtering: DISABLED')
    
    # Print ROI cropping configuration
    if use_roi_crop:
        print(f'\nROI cropping enabled:')
        print(f'  roi_margin: {roi_margin}')
        print(f'  roi_target_size: {roi_target_size}')
    else:
        print(f'\nROI cropping: DISABLED')
    
    # Print MedIM configuration
    print(f'\nModel loading: {"MedIM (recommended)" if use_medim else "Legacy method"}')
    print()
    
    # Find SAM-Med3D root and checkpoint
    print("Checking SAM-Med3D checkpoint...")
    sam3d_root = find_default_sam3d_root()
    checkpoint_path = sam3d_root / 'ckpt' / 'sam_med3d_turbo.pth'
    if not checkpoint_path.exists():
        checkpoint_path = sam3d_root / 'ckpt' / 'SAM-Med3D-turbo.pth'
    if not checkpoint_path.exists():
        print("⚠️  WARNING: No SAM-Med3D checkpoint found! Will use random weights.")
        print(f"   Download from: https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth")
        print(f"   Save to: {sam3d_root / 'ckpt' / 'sam_med3d_turbo.pth'}")
        checkpoint_path = None
    else:
        print(f"✅ Checkpoint found: {checkpoint_path}\n")
    
    # Dry run mode - print what will run and exit
    if dry_run:
        print(f'{"="*80}')
        print('DRY RUN MODE - No training will be executed')
        print(f'{"="*80}\n')
        
        print(f'📊 Total datasets: {len(datasets_to_run)}')
        for i, ds in enumerate(datasets_to_run, 1):
            print(f'   {i}. {ds}')
        
        print(f'\n🔬 Methods to run:')
        methods_count = 0
        if not skip_tabpfn:
            methods_count += 1
            print(f'   ✓ Med3-TabPFN')
        else:
            print(f'   ✗ Med3-TabPFN (skipped)')
            
        if not skip_localpfn:
            methods_count += 1
            print(f'   ✓ Med3-LoCalPFN')
        else:
            print(f'   ✗ Med3-LoCalPFN (skipped)')
            
        if not skip_baselines:
            methods_count += 2
            print(f'   ✓ DenseNet121-3D ({epochs_3d} epochs)')
            print(f'   ✓ ViT-3D ({epochs_3d} epochs)')
        else:
            print(f'   ✗ DenseNet121-3D (skipped)')
            print(f'   ✗ ViT-3D (skipped)')
        
        total_runs = len(datasets_to_run) * methods_count
        print(f'\n📈 Total training runs: {total_runs}')
        print(f'   ({len(datasets_to_run)} datasets × {methods_count} methods)')
        
        print(f'\n📁 Outputs will be saved to: {outputs_base}')
        print(f'\n✅ Dry run complete. Everything looks good!')
        print(f'   Remove --dry-run flag to execute actual training.\n')
        
        return None
    
    results = {}
    
    # ========== PFN Methods ==========
    
    if not skip_tabpfn:
        print(f'\n{"="*80}')
        print('RUNNING: Med3-TabPFN')
        print(f'{"="*80}\n')
        try:
            res_tab = run_multi_tabpfn(
                config_path=config_path,
                dataset_names=datasets_to_run,
                outputs_base_dir=outputs_base,
                checkpoint=checkpoint_path,
                n_splits=n_splits,
                # Pooling
                pooling_strategy=pooling_strategy,
                # ROI cropping parameters
                use_roi_crop=use_roi_crop,
                roi_margin=roi_margin,
                roi_target_size=roi_target_size,
                # Lesion filtering
                lesion_filter=lesion_filter,
            )
            results['tabpfn'] = res_tab
            print('\n[SUCCESS] TabPFN completed')
            print(res_tab['summary_df'])
        except Exception as e:
            print(f'\n[ERROR] TabPFN failed: {e}')
            import traceback
            traceback.print_exc()
            results['tabpfn'] = None
    
    if not skip_localpfn:
        print(f'\n{"="*80}')
        print('RUNNING: Med3-LoCalPFN')
        print(f'{"="*80}\n')
        try:
            res_loc = run_multi_localpfn(
                config_path=config_path,
                dataset_names=datasets_to_run,
                outputs_base_dir=outputs_base,
                checkpoint=checkpoint_path,
                n_splits=n_splits,
                # Pooling
                pooling_strategy=pooling_strategy,
                # ROI cropping parameters
                use_roi_crop=use_roi_crop,
                roi_margin=roi_margin,
                roi_target_size=roi_target_size,
                # LoCalPFN configuration
                local_k=128,
                local_fit_adapter=True,
                local_adapter_epochs=8,
                local_adapter_num_queries=150,
                # Lesion filtering
                lesion_filter=lesion_filter,
            )
            results['localpfn'] = res_loc
            print('\n[SUCCESS] LoCalPFN completed')
            print(res_loc['summary_df'])
        except Exception as e:
            print(f'\n[ERROR] LoCalPFN failed: {e}')
            import traceback
            traceback.print_exc()
            results['localpfn'] = None
    
    # ========== 3D Baselines ==========
    
    if not skip_baselines:
        print(f'\n{"="*80}')
        print('RUNNING: 3D Baselines (DenseNet121 & ViT)')
        print(f'{"="*80}\n')
        
        sam3d_root = find_default_sam3d_root()
        project_root = sam3d_root.parent.parent.resolve()
        
        rows: List[Dict[str, Any]] = []
        
        for ds_key in datasets_to_run:
            print(f'\n--- Processing dataset: {ds_key} ---')
            ds_cfg = cfg['datasets'][ds_key] or {}
            category = ds_cfg.get('category', ds_key)
            ct_name = ds_cfg.get('ct_name', f'ct_{ds_key.upper()}')
            ds_root = _resolve_dataset_root(ds_cfg.get('dataset_root'), 
                                          category=category, 
                                          project_root=project_root)
            
            labels = ds_cfg.get('labels', {}) or {}
            sheet_csv = labels.get('sheet_csv')
            sheet_path = (ds_root / sheet_csv) if sheet_csv else (ds_root / 'sheet.csv')
            dataset_name = labels.get('dataset_name')
            subject_col = labels.get('subject_col', 'Subject')
            label_col = labels.get('label_col', 'Diagnosis_binary')
            case_suffix = labels.get('case_suffix', '_CT')
            
            paths = Sam3DPaths(sam3d_root=sam3d_root, category=category, ct_name=ct_name)
            paths.ensure()
            
            # DenseNet121-3D
            print(f'\n  -> Running DenseNet121-3D on {ds_key}...')
            try:
                d121 = train_eval_densenet121_3d(
                    paths=paths,
                    sheet_csv=sheet_path,
                    dataset_name=dataset_name,
                    subject_col=subject_col,
                    label_col=label_col,
                    case_suffix=case_suffix,
                    dataset_root=ds_root,
                    epochs=epochs_3d,
                    device=None,
                    lesion_filter=lesion_filter,
                )
                er = d121['eval']
                rows.append({
                    'dataset': ds_key, 
                    'category': category, 
                    'ct_name': ct_name, 
                    'method': 'densenet121_3d',
                    'accuracy': er['acc'], 
                    'macro_f1': er['macro_f1'], 
                    'roc_auc': er.get('roc_auc'),
                    'out_dir': str(d121.get('out_dir', ''))
                })
                print(f'     [SUCCESS] DenseNet121-3D: acc={er["acc"]:.4f}')
            except Exception as e:
                print(f'     [ERROR] DenseNet121-3D failed: {e}')
                rows.append({
                    'dataset': ds_key, 
                    'category': category, 
                    'ct_name': ct_name, 
                    'method': 'densenet121_3d',
                    'accuracy': None, 
                    'macro_f1': None, 
                    'roc_auc': None, 
                    'error': str(e)
                })
            
            # ViT-3D
            print(f'\n  -> Running ViT-3D on {ds_key}...')
            try:
                vit = train_eval_vit_3d(
                    paths=paths,
                    sheet_csv=sheet_path,
                    dataset_name=dataset_name,
                    subject_col=subject_col,
                    label_col=label_col,
                    case_suffix=case_suffix,
                    dataset_root=ds_root,
                    epochs=epochs_3d,
                    device=None,
                    lesion_filter=lesion_filter,
                )
                er = vit['eval']
                rows.append({
                    'dataset': ds_key, 
                    'category': category, 
                    'ct_name': ct_name, 
                    'method': 'vit3d',
                    'accuracy': er['acc'], 
                    'macro_f1': er['macro_f1'], 
                    'roc_auc': er.get('roc_auc'),
                    'out_dir': str(vit.get('out_dir', ''))
                })
                print(f'     [SUCCESS] ViT-3D: acc={er["acc"]:.4f}')
            except Exception as e:
                print(f'     [ERROR] ViT-3D failed: {e}')
                rows.append({
                    'dataset': ds_key, 
                    'category': category, 
                    'ct_name': ct_name, 
                    'method': 'vit3d',
                    'accuracy': None, 
                    'macro_f1': None, 
                    'roc_auc': None, 
                    'error': str(e)
                })
        
        df_baselines = pd.DataFrame(rows)
        results['baselines'] = df_baselines
        print('\n[SUCCESS] 3D baselines completed')
        print(df_baselines)
    
    # ========== Combined Summary ==========
    
    print(f'\n{"="*80}')
    print('GENERATING COMBINED SUMMARY')
    print(f'{"="*80}\n')
    
    dfs_to_combine = []
    
    if not skip_tabpfn and results.get('tabpfn'):
        dfs_to_combine.append(results['tabpfn']['summary_df'])
    
    if not skip_localpfn and results.get('localpfn'):
        dfs_to_combine.append(results['localpfn']['summary_df'])
    
    if not skip_baselines and results.get('baselines') is not None:
        dfs_to_combine.append(results['baselines'])
    
    if dfs_to_combine:
        combined = pd.concat(dfs_to_combine, ignore_index=True, sort=False)
        
        # Save combined summary
        outputs_base.mkdir(parents=True, exist_ok=True)
        out_csv = outputs_base / 'combined_benchmarks_summary.csv'
        combined.to_csv(out_csv, index=False)
        
        print(f'[SUCCESS] Combined summary saved to: {out_csv}')
        print('\nFinal Results:')
        print(combined.to_string())
        
        # ========== Compute Average Scores Per Method ==========
        print(f'\n{"="*80}')
        print('COMPUTING AVERAGE SCORES PER METHOD')
        print(f'{"="*80}\n')
        
        # Only compute averages for numeric columns
        numeric_cols = ['accuracy', 'macro_f1', 'roc_auc']
        
        # Filter to only successful runs (non-null accuracy)
        successful_runs = combined[combined['accuracy'].notna()].copy()
        
        if len(successful_runs) > 0:
            # Compute averages per method
            avg_scores = successful_runs.groupby('method')[numeric_cols].agg(['mean', 'std', 'count'])
            
            # Flatten multi-index columns
            avg_scores.columns = ['_'.join(col).strip() for col in avg_scores.columns.values]
            avg_scores = avg_scores.reset_index()
            
            # Rename for clarity
            avg_scores.rename(columns={
                'accuracy_mean': 'avg_accuracy',
                'accuracy_std': 'std_accuracy',
                'accuracy_count': 'n_datasets',
                'macro_f1_mean': 'avg_macro_f1',
                'macro_f1_std': 'std_macro_f1',
                'macro_f1_count': 'n_datasets_f1',
                'roc_auc_mean': 'avg_roc_auc',
                'roc_auc_std': 'std_roc_auc',
                'roc_auc_count': 'n_datasets_auc'
            }, inplace=True)
            
            # Keep only useful columns
            avg_scores = avg_scores[['method', 'n_datasets', 'avg_accuracy', 'std_accuracy', 
                                     'avg_macro_f1', 'std_macro_f1', 'avg_roc_auc', 'std_roc_auc']]
            
            # Sort by average ROC AUC (best first)
            avg_scores = avg_scores.sort_values('avg_roc_auc', ascending=False)
            
            # Save average scores
            avg_csv = outputs_base / 'average_scores_per_method.csv'
            avg_scores.to_csv(avg_csv, index=False)
            
            print(f'[SUCCESS] Average scores saved to: {avg_csv}')
            print('\nAverage Scores Per Method:')
            print(avg_scores.to_string(index=False))
            print(f'\n(Sorted by ROC AUC, higher is better)')
        else:
            print('[WARNING] No successful runs to compute averages')
        
        return combined
    else:
        print('[WARNING] No results to combine')
        return None


def main():
    print("[DEBUG] Script started, parsing arguments...", flush=True)
    parser = argparse.ArgumentParser(
        description='Run Experiment 1: Head-to-head Benchmarks'
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
        default='results/experiment1',
        help='Output directory for results (default: results/experiment1)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=None,
        help='Specific datasets to run (default: all in config)'
    )
    parser.add_argument(
        '--pooling-strategy',
        type=str,
        choices=['avg', 'multiscale', 'percentile'],
        default='avg',
        help='Pooling strategy for embeddings (choices: avg, multiscale, percentile). Default: avg'
    )
    parser.add_argument(
        '--skip-tabpfn',
        action='store_true',
        help='Skip TabPFN method'
    )
    parser.add_argument(
        '--skip-localpfn',
        action='store_true',
        help='Skip LoCalPFN method'
    )
    parser.add_argument(
        '--skip-baselines',
        action='store_true',
        help='Skip 3D baseline methods'
    )
    parser.add_argument(
        '--epochs-3d',
        type=int,
        default=4,
        help='Number of epochs for 3D models (default: 4)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what will be executed without running (for verification)'
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
    # ROI cropping arguments
    parser.add_argument(
        '--use-roi-crop',
        action='store_true',
        help='Enable ROI cropping (tumor-centered volumes)'
    )
    parser.add_argument(
        '--roi-margin',
        type=int,
        default=10,
        help='Margin around tumor for ROI cropping in voxels (default: 10)'
    )
    parser.add_argument(
        '--roi-target-size',
        type=int,
        default=128,
        help='Target size after ROI cropping (default: 128)'
    )
    # MedIM integration
    parser.add_argument(
        '--use-medim',
        action='store_true',
        default=True,
        help='Use MedIM for model loading (default: True, recommended)'
    )
    parser.add_argument(
        '--no-medim',
        action='store_true',
        help='Disable MedIM, use legacy model loading'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Number of k-fold cross-validation splits (default: 5)'
    )
    
    args = parser.parse_args()
    print("[DEBUG] Arguments parsed successfully", flush=True)
    
    # Setup paths
    print("[DEBUG] Adding repo root to sys.path...", flush=True)
    repo_root = _add_repo_root_to_sys_path()
    print("[DEBUG] Repo root added", flush=True)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    
    outputs_base = Path(args.output_dir)
    if not outputs_base.is_absolute():
        outputs_base = repo_root / outputs_base
    
    # Apply filter presets or use individual parameters
    min_voxels = args.min_voxels
    min_dimension = args.min_dimension
    min_density = args.min_density
    
    if args.filter_preset:
        print(f'[INFO] Using filter preset: {args.filter_preset}')
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
    
    # Determine MedIM usage
    use_medim = args.use_medim and not args.no_medim
    
    # Run experiment
    print(f"[DEBUG] Calling run_experiment with config: {config_path}", flush=True)
    try:
        results = run_experiment(
            config_path=config_path,
            outputs_base=outputs_base,
            dataset_filter=args.datasets,
            skip_tabpfn=args.skip_tabpfn,
            skip_localpfn=args.skip_localpfn,
            skip_baselines=args.skip_baselines,
            epochs_3d=args.epochs_3d,
            dry_run=args.dry_run,
            n_splits=args.n_splits,
            # Pooling
            pooling_strategy=args.pooling_strategy,
            # ROI cropping parameters
            use_roi_crop=args.use_roi_crop,
            roi_margin=args.roi_margin,
            roi_target_size=args.roi_target_size,
            # MedIM integration
            use_medim=use_medim,
            # Lesion filtering
            min_voxels=min_voxels,
            min_dimension=min_dimension,
            min_density=min_density,
        )
        
        print(f'\n{"="*80}')
        print('EXPERIMENT COMPLETED SUCCESSFULLY')
        print(f'{"="*80}\n')
        
        return 0
        
    except Exception as e:
        print(f'\n{"="*80}')
        print('EXPERIMENT FAILED')
        print(f'{"="*80}')
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
