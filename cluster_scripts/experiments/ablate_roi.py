#!/usr/bin/env python3
"""
Ablation Study: ROI Margin Effects

This script runs ablation experiments to find optimal ROI margin:
- Margin = 5:  Tight crop (minimal context)
- Margin = 10: Standard (balanced)
- Margin = 20: Loose crop (more context)

All experiments run on GPU with TabPFN.
"""

from pathlib import Path
import sys
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from med3pipe.pipelines import run_multi_tabpfn


def main():
    # Configuration
    config_path = project_root / "configs" / "datasets.yaml"
    results_dir = project_root / "results" / "roi_ablation"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"ROI MARGIN ABLATION STUDY")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")
    
    # Margins to test
    margins = [5, 10, 15, 20]
    
    all_results = {}
    
    for margin in margins:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: ROI Margin = {margin} voxels")
        print(f"{'='*60}\n")
        
        try:
            results = run_multi_tabpfn(
                config_path=config_path,
                
                # ROI cropping with varying margin
                use_roi_crop=True,
                roi_margin=margin,
                roi_target_size=128,
                
                # Model parameters
                model_type="vit_b_ori",
                device=device,
                
                # Feature extraction
                skip_existing_embeddings=False,
                
                # TabPFN parameters
                n_components_max=500,
                random_state=42,
                
                # Output
                outputs_base_dir=results_dir / f"margin_{margin}",
                save_summary=True,
                summary_path=results_dir / f"margin_{margin}_summary.csv",
            )
            
            all_results[margin] = results
            print(f"\n✅ Margin {margin} experiment completed")
            
        except Exception as e:
            print(f"\n❌ Margin {margin} experiment failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare all margins
    print("\n" + "="*60)
    print("ABLATION COMPARISON")
    print("="*60 + "\n")
    
    try:
        import pandas as pd
        
        # Load all summaries
        dfs = {}
        for margin in margins:
            summary_path = results_dir / f"margin_{margin}_summary.csv"
            if summary_path.exists():
                dfs[margin] = pd.read_csv(summary_path)
        
        if not dfs:
            print("⚠️  No results to compare")
            return
        
        # Build comparison table
        datasets = dfs[margins[0]]['dataset'].unique()
        
        comparison_rows = []
        for dataset in datasets:
            row = {'dataset': dataset}
            for margin in margins:
                if margin in dfs:
                    df = dfs[margin]
                    dataset_row = df[df['dataset'] == dataset]
                    if not dataset_row.empty:
                        row[f'acc_m{margin}'] = dataset_row['accuracy'].values[0]
                        row[f'auc_m{margin}'] = dataset_row['roc_auc'].values[0]
            comparison_rows.append(row)
        
        comparison = pd.DataFrame(comparison_rows)
        
        # Save comparison
        comparison_path = results_dir / "ablation_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        
        print(comparison.to_string(index=False))
        print(f"\n✅ Comparison saved to: {comparison_path}")
        
        # Find best margin per dataset
        print("\n" + "-"*60)
        print("BEST MARGIN PER DATASET (by AUC)")
        print("-"*60)
        
        for _, row in comparison.iterrows():
            dataset = row['dataset']
            auc_cols = [c for c in row.index if c.startswith('auc_m')]
            if auc_cols:
                best_col = max(auc_cols, key=lambda c: row[c] if pd.notna(row[c]) else -1)
                best_margin = int(best_col.split('m')[1])
                best_auc = row[best_col]
                print(f"{dataset}: Margin {best_margin} (AUC = {best_auc:.4f})")
        
        # Average across datasets
        print("\n" + "-"*60)
        print("AVERAGE PERFORMANCE BY MARGIN")
        print("-"*60)
        
        for margin in margins:
            auc_col = f'auc_m{margin}'
            if auc_col in comparison.columns:
                avg_auc = comparison[auc_col].mean()
                print(f"Margin {margin:2d}: AUC = {avg_auc:.4f}")
        
    except Exception as e:
        print(f"⚠️  Could not generate comparison: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)
    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
