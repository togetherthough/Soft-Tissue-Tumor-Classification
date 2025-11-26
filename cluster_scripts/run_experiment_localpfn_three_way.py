#!/usr/bin/env python3
"""
Experiment: LoCalPFN - Three-Way Comparison

This script runs a comprehensive comparison experiment:
1. Baseline: Full-volume, no lesion filtering
2. Filtered Baseline: Full-volume with lesion size filtering
3. ROI-Cropped: Adaptive crop/pad (tumor-centered volumes)

All experiments run on GPU and save results for comparison.
"""

from pathlib import Path
import sys
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from med3pipe.pipelines import run_multi_localpfn
from med3pipe.tabular.lesion_filter import LesionSizeFilter


def main():
    # Configuration
    config_path = project_root / "configs" / "datasets.yaml"
    results_dir = project_root / "results" / "three_way_comparison_localpfn"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"THREE-WAY COMPARISON EXPERIMENT - LoCalPFN")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")
    
    # Lesion filter configuration for experiments 2 & 3
    lesion_filter = LesionSizeFilter(
        min_voxels=100,      # Minimum 100 voxels
        min_dimension=3,     # At least 3 voxels in each dimension
        min_density=0.1,     # At least 10% of bounding box should be lesion
    )
    
    # LoCalPFN configuration
    local_k = 128
    local_fit_adapter = True
    local_adapter_epochs = 10
    
    # Experiment 1: Baseline (Full-Volume, No Filtering)
    print("\n" + "="*60)
    print("EXPERIMENT 1: BASELINE (Full-Volume, No Filtering)")
    print("="*60 + "\n")
    
    try:
        results_baseline = run_multi_localpfn(
            config_path=config_path,
            
            # Full-volume preprocessing, NO filtering
            use_roi_crop=False,
            
            # Model parameters
            model_type="vit_b_ori",
            device=device,
            
            # Feature extraction
            skip_existing_embeddings=False,  # Reuse if available
            
            # LoCalPFN parameters
            n_components_max=500,
            random_state=42,
            local_k=local_k,
            local_metric="euclidean",
            local_fit_adapter=local_fit_adapter,
            local_adapter_epochs=local_adapter_epochs,
            local_adapter_lr=5e-2,
            local_adapter_weight_decay=0.0,
            local_adapter_num_queries=1000,
            
            # Output
            outputs_base_dir=results_dir / "baseline",
            save_summary=True,
            summary_path=results_dir / "baseline_summary.csv",
        )
        
        print(f"\n✅ Baseline experiment completed")
        print(f"   Results: {results_dir / 'baseline_summary.csv'}")
        
    except Exception as e:
        print(f"\n❌ Baseline experiment failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Experiment 2: Filtered Baseline (Full-Volume with Lesion Filtering)
    print("\n" + "="*60)
    print("EXPERIMENT 2: FILTERED BASELINE (Full-Volume + Lesion Filtering)")
    print("="*60 + "\n")
    print(f"Lesion filter: min_voxels={lesion_filter.min_voxels}, "
          f"min_dimension={lesion_filter.min_dimension}, min_density={lesion_filter.min_density}")
    print()
    
    try:
        results_filtered = run_multi_localpfn(
            config_path=config_path,
            
            # Full-volume preprocessing with lesion filtering
            use_roi_crop=False,
            min_voxels=lesion_filter.min_voxels,
            min_dimension=lesion_filter.min_dimension,
            min_density=lesion_filter.min_density,
            
            # Model parameters
            model_type="vit_b_ori",
            device=device,
            
            # Feature extraction
            skip_existing_embeddings=False,  # Reuse if available
            
            # LoCalPFN parameters
            n_components_max=500,
            random_state=42,
            local_k=local_k,
            local_metric="euclidean",
            local_fit_adapter=local_fit_adapter,
            local_adapter_epochs=local_adapter_epochs,
            local_adapter_lr=5e-2,
            local_adapter_weight_decay=0.0,
            local_adapter_num_queries=1000,
            
            # Output
            outputs_base_dir=results_dir / "filtered_baseline",
            save_summary=True,
            summary_path=results_dir / "filtered_baseline_summary.csv",
        )
        
        print(f"\n✅ Filtered baseline experiment completed")
        print(f"   Results: {results_dir / 'filtered_baseline_summary.csv'}")
        
    except Exception as e:
        print(f"\n❌ Filtered baseline experiment failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Experiment 3: ROI-Cropped (Adaptive Crop/Pad)
    print("\n" + "="*60)
    print("EXPERIMENT 3: ROI-CROPPED (Adaptive Crop/Pad)")
    print("="*60 + "\n")
    
    try:
        results_roi = run_multi_localpfn(
            config_path=config_path,
            
            # ROI-centric preprocessing (proposed)
            use_roi_crop=True,
            roi_margin=10,          # 10 voxels context
            roi_target_size=128,    # Final volume size
            
            # Model parameters
            model_type="vit_b_ori",
            device=device,
            
            # Feature extraction (force re-extraction for ROI data)
            skip_existing_embeddings=False,
            
            # LoCalPFN parameters
            n_components_max=500,
            random_state=42,
            local_k=local_k,
            local_metric="euclidean",
            local_fit_adapter=local_fit_adapter,
            local_adapter_epochs=local_adapter_epochs,
            local_adapter_lr=5e-2,
            local_adapter_weight_decay=0.0,
            local_adapter_num_queries=1000,
            
            # Output
            outputs_base_dir=results_dir / "roi_cropped",
            save_summary=True,
            summary_path=results_dir / "roi_cropped_summary.csv",
        )
        
        print(f"\n✅ ROI-cropped experiment completed")
        print(f"   Results: {results_dir / 'roi_cropped_summary.csv'}")
        
    except Exception as e:
        print(f"\n❌ ROI-cropped experiment failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60 + "\n")
    
    try:
        import pandas as pd
        
        # Load all three results
        df_baseline = pd.read_csv(results_dir / "baseline_summary.csv")
        df_filtered = pd.read_csv(results_dir / "filtered_baseline_summary.csv")
        df_roi = pd.read_csv(results_dir / "roi_cropped_summary.csv")
        
        # Merge on dataset
        comparison = pd.merge(
            df_baseline[['dataset', 'accuracy', 'macro_f1', 'roc_auc']],
            df_filtered[['dataset', 'accuracy', 'macro_f1', 'roc_auc']],
            on='dataset',
            suffixes=('_baseline', '_filtered')
        )
        comparison = pd.merge(
            comparison,
            df_roi[['dataset', 'accuracy', 'macro_f1', 'roc_auc']],
            on='dataset'
        )
        comparison.rename(columns={
            'accuracy': 'accuracy_roi',
            'macro_f1': 'macro_f1_roi',
            'roc_auc': 'roc_auc_roi'
        }, inplace=True)
        
        # Calculate improvements (vs baseline)
        comparison['accuracy_delta_filtered'] = comparison['accuracy_filtered'] - comparison['accuracy_baseline']
        comparison['accuracy_delta_roi'] = comparison['accuracy_roi'] - comparison['accuracy_baseline']
        comparison['f1_delta_filtered'] = comparison['macro_f1_filtered'] - comparison['macro_f1_baseline']
        comparison['f1_delta_roi'] = comparison['macro_f1_roi'] - comparison['macro_f1_baseline']
        comparison['auc_delta_filtered'] = comparison['roc_auc_filtered'] - comparison['roc_auc_baseline']
        comparison['auc_delta_roi'] = comparison['roc_auc_roi'] - comparison['roc_auc_baseline']
        
        # Save comparison
        comparison_path = results_dir / "three_way_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        
        print(comparison.to_string(index=False))
        print(f"\n✅ Comparison saved to: {comparison_path}")
        
        # Summary statistics
        print("\n" + "-"*60)
        print("AVERAGE IMPROVEMENTS vs. BASELINE")
        print("-"*60)
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
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
