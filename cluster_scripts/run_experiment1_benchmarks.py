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

import torch
torch.set_num_threads(1)

import yaml
import pandas as pd


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
):
    """
    Run the full benchmark experiment
    
    Args:
        config_path: Path to datasets.yaml config
        outputs_base: Base directory for outputs
        dataset_filter: Optional list of dataset names to run (None = all)
        skip_tabpfn: Skip TabPFN method
        skip_localpfn: Skip LoCalPFN method
        skip_baselines: Skip 3D baseline methods
        epochs_3d: Number of epochs for 3D models training
    """
    
    # Import after path setup
    from med3pipe.pipelines import run_multi_tabpfn, run_multi_localpfn
    from med3pipe.data.prepare import Sam3DPaths, find_default_sam3d_root
    from med3pipe.vision.v3d import train_eval_densenet121_3d, train_eval_vit_3d
    
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
    print(f'Datasets to run: {datasets_to_run}\n')
    
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
                local_k=8,
                local_fit_adapter=True,
                local_adapter_epochs=8,
                local_adapter_num_queries=150,
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
        
        return combined
    else:
        print('[WARNING] No results to combine')
        return None


def main():
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
    
    args = parser.parse_args()
    
    # Setup paths
    repo_root = _add_repo_root_to_sys_path()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    
    outputs_base = Path(args.output_dir)
    if not outputs_base.is_absolute():
        outputs_base = repo_root / outputs_base
    
    # Run experiment
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
