#!/usr/bin/env python3
"""
Preprocessing Comparison Experiment for Cluster

Compares three preprocessing strategies for SAM-Med3D feature extraction:
1. Baseline: Full-volume, no lesion filtering
2. Filtered Baseline: Full-volume with lesion size filtering
3. ROI-Cropped: Adaptive crop/pad (tumor-centered volumes)

All experiments use TabPFN with k-fold cross-validation.

Usage:
    python cluster_scripts/experiments/compare_preprocessing.py \
        --config configs/datasets_cluster.yaml \
        --output-dir results/preprocessing_comparison

Based on: notebooks/experiments/Preprocessing_Comparisons_Local.ipynb
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch


def setup_paths():
    """Setup project paths."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare preprocessing strategies for SAM-Med3D embeddings"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to datasets YAML config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/preprocessing_comparison",
        help="Directory to save results"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Specific datasets to run (default: all from config)"
    )
    parser.add_argument(
        "--pooling-strategy",
        type=str,
        choices=["avg", "multiscale", "percentile"],
        default="avg",
        help="Pooling strategy for embeddings (choices: avg, multiscale, percentile). Default: avg"
    )
    parser.add_argument(
        "--roi-margin",
        type=int,
        default=30,
        help="ROI margin in voxels (default: 30)"
    )
    parser.add_argument(
        "--roi-target-size",
        type=int,
        default=128,
        help="ROI target volume size (default: 128)"
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=300,
        help="Minimum voxels for lesion filtering (default: 300)"
    )
    parser.add_argument(
        "--min-dimension",
        type=int,
        default=5,
        help="Minimum dimension for lesion filtering (default: 5)"
    )
    parser.add_argument(
        "--min-density",
        type=float,
        default=0.1,
        help="Minimum density for lesion filtering (default: 0.1)"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-validation splits (default: 5)"
    )
    parser.add_argument(
        "--n-components-max",
        type=int,
        default=500,
        help="Maximum PCA components (default: 500)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline experiment"
    )
    parser.add_argument(
        "--skip-filtered",
        action="store_true",
        help="Skip filtered baseline experiment"
    )
    parser.add_argument(
        "--skip-roi",
        action="store_true",
        help="Skip ROI-cropped experiment"
    )
    
    return parser.parse_args()


def load_sam_model(project_root: Path, device: str):
    """Load SAM-Med3D model using medim."""
    import medim
    
    checkpoint_path = project_root / "sam-med3d" / "ckpt" / "sam_med3d_turbo.pth"
    ckpt_dir = checkpoint_path.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Download checkpoint if not present
    if not checkpoint_path.exists():
        url = "https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth"
        print(f"Checkpoint missing. Downloading to: {checkpoint_path}")
        try:
            import shutil
            import urllib.request
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as r, open(checkpoint_path, 'wb') as f:
                shutil.copyfileobj(r, f)
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            print(f"✅ Download complete: {size_mb:.1f} MB")
        except Exception as e:
            print(f"❌ Failed to download checkpoint: {e}")
            raise
    else:
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"✅ Checkpoint present: {checkpoint_path} ({size_mb:.1f} MB)")
    
    print("\n📦 Loading SAM-Med3D model via medim...")
    sam_model = medim.create_model(
        "SAM-Med3D",
        pretrained=True,
        checkpoint_path=str(checkpoint_path)
    ).to(device)
    sam_model.eval()
    print("✅ SAM-Med3D model loaded successfully!")
    print(f"   Model device: {next(sam_model.parameters()).device}")
    
    return sam_model


def run_baseline_experiment(
    config_path: Path,
    output_dir: Path,
    sam_model,
    device: str,
    n_splits: int,
    n_components_max: int,
    random_state: int,
    pooling_strategy: str = 'avg',
):
    """Run baseline experiment (full-volume, no filtering)."""
    from med3pipe.pipelines import run_multi_tabpfn
    
    print("\n" + "=" * 70)
    print("BASELINE (Full-Volume, No Filtering)")
    print("=" * 70 + "\n")
    
    results_dir = output_dir / "baseline"
    summary_path = output_dir / "baseline_summary.csv"
    
    try:
        results = run_multi_tabpfn(
            config_path=config_path,
            
            # Full-volume preprocessing, NO filtering
            use_roi_crop=False,
            
            # Model parameters
            model=sam_model,
            device=device,
            
            # Feature extraction
            skip_existing_embeddings=False,
            
            # Pooling
            pooling_strategy=pooling_strategy,
            
            # TabPFN parameters
            n_components_max=n_components_max,
            random_state=random_state,
            n_splits=n_splits,
            
            # Output
            outputs_base_dir=results_dir,
            save_summary=True,
            summary_path=summary_path,
        )
        
        print(f"\n✅ Baseline experiment completed")
        print(f"   Results: {summary_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Baseline experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_filtered_experiment(
    config_path: Path,
    output_dir: Path,
    sam_model,
    device: str,
    n_splits: int,
    n_components_max: int,
    random_state: int,
    min_voxels: int,
    min_dimension: int,
    min_density: float,
    pooling_strategy: str = 'avg',
):
    """Run filtered baseline experiment (full-volume with lesion filtering)."""
    from med3pipe.pipelines import run_multi_tabpfn
    
    print("\n" + "=" * 70)
    print("FILTERED BASELINE (Full-Volume + Lesion Filtering)")
    print("=" * 70 + "\n")
    
    print(f"Lesion filter: min_voxels={min_voxels}, "
          f"min_dimension={min_dimension}, min_density={min_density}\n")
    
    results_dir = output_dir / "filtered_baseline"
    summary_path = output_dir / "filtered_baseline_summary.csv"
    
    try:
        results = run_multi_tabpfn(
            config_path=config_path,
            
            # Full-volume preprocessing with lesion filtering
            use_roi_crop=False,
            min_voxels=min_voxels,
            min_dimension=min_dimension,
            min_density=min_density,
            
            # Model parameters
            model=sam_model,
            device=device,
            
            # Feature extraction
            skip_existing_embeddings=False,
            
            # Pooling
            pooling_strategy=pooling_strategy,
            
            # TabPFN parameters
            n_components_max=n_components_max,
            random_state=random_state,
            n_splits=n_splits,
            
            # Output
            outputs_base_dir=results_dir,
            save_summary=True,
            summary_path=summary_path,
        )
        
        print(f"\n✅ Filtered baseline experiment completed")
        print(f"   Results: {summary_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Filtered baseline experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_roi_experiment(
    config_path: Path,
    output_dir: Path,
    sam_model,
    device: str,
    n_splits: int,
    n_components_max: int,
    random_state: int,
    roi_margin: int,
    roi_target_size: int,
    pooling_strategy: str = 'avg',
):
    """Run ROI-cropped experiment (adaptive crop/pad)."""
    from med3pipe.pipelines import run_multi_tabpfn
    
    print("\n" + "=" * 70)
    print("ROI-CROPPED (Adaptive Crop/Pad)")
    print("=" * 70 + "\n")
    
    print(f"ROI settings: margin={roi_margin}, target_size={roi_target_size}\n")
    
    results_dir = output_dir / "roi_cropped"
    summary_path = output_dir / "roi_cropped_summary.csv"
    
    try:
        results = run_multi_tabpfn(
            config_path=config_path,
            
            # ROI-centric preprocessing
            use_roi_crop=True,
            roi_margin=roi_margin,
            roi_target_size=roi_target_size,
            
            # Model parameters
            model=sam_model,
            device=device,
            
            # Feature extraction (force re-extraction for ROI data)
            skip_existing_embeddings=True,
            
            # Pooling
            pooling_strategy=pooling_strategy,
            
            # TabPFN parameters
            n_components_max=n_components_max,
            random_state=random_state,
            n_splits=n_splits,
            
            # Output
            outputs_base_dir=results_dir,
            save_summary=True,
            summary_path=summary_path,
        )
        
        print(f"\n✅ ROI-cropped experiment completed")
        print(f"   Results: {summary_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ ROI-cropped experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_comparison(output_dir: Path):
    """Generate comparison of all three preprocessing strategies."""
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPARISON")
    print("=" * 70 + "\n")
    
    baseline_path = output_dir / "baseline_summary.csv"
    filtered_path = output_dir / "filtered_baseline_summary.csv"
    roi_path = output_dir / "roi_cropped_summary.csv"
    
    # Check which files exist
    available = []
    if baseline_path.exists():
        available.append(("baseline", pd.read_csv(baseline_path)))
    if filtered_path.exists():
        available.append(("filtered", pd.read_csv(filtered_path)))
    if roi_path.exists():
        available.append(("roi", pd.read_csv(roi_path)))
    
    if len(available) < 2:
        print("⚠️  Need at least 2 experiments to compare. Skipping comparison.")
        return None
    
    # Build comparison dataframe
    comparison = None
    for name, df in available:
        df_subset = df[['dataset', 'accuracy', 'macro_f1', 'roc_auc']].copy()
        df_subset.columns = ['dataset'] + [f'{col}_{name}' for col in ['accuracy', 'macro_f1', 'roc_auc']]
        
        if comparison is None:
            comparison = df_subset
        else:
            comparison = pd.merge(comparison, df_subset, on='dataset', how='outer')
    
    # Calculate deltas if baseline exists
    if baseline_path.exists():
        for name, _ in available:
            if name != 'baseline':
                for metric in ['accuracy', 'macro_f1', 'roc_auc']:
                    baseline_col = f'{metric}_baseline'
                    current_col = f'{metric}_{name}'
                    delta_col = f'{metric}_delta_{name}'
                    
                    if baseline_col in comparison.columns and current_col in comparison.columns:
                        comparison[delta_col] = comparison[current_col] - comparison[baseline_col]
    
    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = output_dir / f"preprocessing_comparison_{timestamp}.csv"
    latest_path = output_dir / "preprocessing_comparison_latest.csv"
    
    comparison.to_csv(comparison_path, index=False)
    comparison.to_csv(latest_path, index=False)
    
    print("Results per dataset:")
    print("-" * 70)
    print(comparison.to_string(index=False))
    print()
    
    # Summary statistics
    print("\n" + "-" * 70)
    print("SUMMARY STATISTICS")
    print("-" * 70)
    
    for name, _ in available:
        print(f"\n{name.upper()}:")
        for metric in ['accuracy', 'macro_f1', 'roc_auc']:
            col = f'{metric}_{name}'
            if col in comparison.columns:
                mean_val = comparison[col].mean()
                std_val = comparison[col].std()
                print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Improvements vs baseline
    if baseline_path.exists():
        print("\n" + "-" * 70)
        print("AVERAGE IMPROVEMENTS vs. BASELINE")
        print("-" * 70)
        
        for name, _ in available:
            if name != 'baseline':
                print(f"\n{name.upper()}:")
                for metric in ['accuracy', 'macro_f1', 'roc_auc']:
                    delta_col = f'{metric}_delta_{name}'
                    if delta_col in comparison.columns:
                        delta_mean = comparison[delta_col].mean()
                        print(f"  {metric}: {delta_mean:+.4f}")
    
    print(f"\n✅ Comparison saved to: {comparison_path}")
    print(f"   Latest symlink: {latest_path}")
    
    return comparison


def main():
    """Main entry point."""
    args = parse_args()
    project_root = setup_paths()
    
    print("=" * 70)
    print("PREPROCESSING COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup paths
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Device setup - GPU required
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ CUDA is not available! This experiment requires GPU.\n"
            f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}\n"
            "   Ensure SLURM allocated GPU and CUDA module is loaded."
        )
    
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load SAM model
    sam_model = load_sam_model(project_root, device)
    
    # Run experiments
    experiments_run = 0
    experiments_passed = 0
    
    if not args.skip_baseline:
        experiments_run += 1
        if run_baseline_experiment(
            config_path=config_path,
            output_dir=output_dir,
            sam_model=sam_model,
            device=device,
            n_splits=args.n_splits,
            n_components_max=args.n_components_max,
            random_state=args.random_state,
            pooling_strategy=args.pooling_strategy,
        ):
            experiments_passed += 1
    
    if not args.skip_filtered:
        experiments_run += 1
        if run_filtered_experiment(
            config_path=config_path,
            output_dir=output_dir,
            sam_model=sam_model,
            device=device,
            n_splits=args.n_splits,
            n_components_max=args.n_components_max,
            random_state=args.random_state,
            min_voxels=args.min_voxels,
            min_dimension=args.min_dimension,
            min_density=args.min_density,
            pooling_strategy=args.pooling_strategy,
        ):
            experiments_passed += 1
    
    if not args.skip_roi:
        experiments_run += 1
        if run_roi_experiment(
            config_path=config_path,
            output_dir=output_dir,
            sam_model=sam_model,
            device=device,
            n_splits=args.n_splits,
            n_components_max=args.n_components_max,
            random_state=args.random_state,
            roi_margin=args.roi_margin,
            roi_target_size=args.roi_target_size,
            pooling_strategy=args.pooling_strategy,
        ):
            experiments_passed += 1
    
    # Generate comparison
    if experiments_passed >= 2:
        generate_comparison(output_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiments: {experiments_passed}/{experiments_run} passed")
    print(f"Results saved to: {output_dir}")
    print()
    
    return 0 if experiments_passed == experiments_run else 1


if __name__ == "__main__":
    sys.exit(main())
