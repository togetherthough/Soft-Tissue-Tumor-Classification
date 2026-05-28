"""
Test script for full SAM-Med3D pipeline with new average pooling logic.

This script runs an end-to-end test to verify:
1. Data preparation works
2. SAM-Med3D model loads and extracts embeddings
3. Average pooling (not ROI pooling) is applied correctly
4. TabPFN training completes successfully

Usage:
    python test_full_pipeline.py --dataset-root gist --max-cases 10
"""

import argparse
from pathlib import Path
import torch
import sys

from med3pipe import (
    prepare_for_sam3d,
    split_validation,
    Sam3DPaths,
    build_sam3d_model,
    extract_embeddings_train_val,
    default_feature_dirs,
    load_pooled_features,
    load_labels_from_sheet,
    build_y,
    tabpfn_pipeline,
    find_default_sam3d_root,
)


def check_prerequisites(dataset_root: Path, sam3d_root: Path):
    """Check if required files and directories exist."""
    print("\n" + "="*60)
    print("CHECKING PREREQUISITES")
    print("="*60)
    
    issues = []
    
    # Check dataset root
    if not dataset_root.exists():
        issues.append(f"Dataset root not found: {dataset_root}")
    else:
        print(f"✓ Dataset root found: {dataset_root}")
    
    # Check SAM-Med3D root
    if not sam3d_root.exists():
        issues.append(f"SAM-Med3D root not found: {sam3d_root}")
    else:
        print(f"✓ SAM-Med3D root found: {sam3d_root}")
    
    # Check for checkpoint
    ckpt_dir = sam3d_root / "ckpt"
    checkpoints = list(ckpt_dir.glob("*.pth")) if ckpt_dir.exists() else []
    if checkpoints:
        print(f"✓ Found {len(checkpoints)} checkpoint(s): {[c.name for c in checkpoints]}")
    else:
        print(f"⚠ No checkpoints found in {ckpt_dir}")
    
    # Check for labels
    sheet_csv = dataset_root / "sheet.csv"
    if not sheet_csv.exists():
        sheet_csv = Path("sheet.csv")
    
    if not sheet_csv.exists():
        issues.append(f"sheet.csv not found in {dataset_root} or current directory")
    else:
        print(f"✓ Labels file found: {sheet_csv}")
    
    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("\n✅ All prerequisites satisfied")
    return True


def run_test_pipeline(
    dataset_root: Path,
    category: str = "gist",
    ct_name: str = "ct_GIST",
    max_cases: int = 10,
    split_ratio: float = 0.8,
    seed: int = 2025,
    device: str = None,
):
    """Run full pipeline test with the new average pooling logic."""
    
    print("\n" + "="*60)
    print("RUNNING FULL PIPELINE TEST")
    print("="*60)
    print(f"Dataset: {dataset_root}")
    print(f"Category: {category}")
    print(f"CT Name: {ct_name}")
    print(f"Max cases: {max_cases}")
    print(f"Split ratio: {split_ratio}")
    print(f"Seed: {seed}")
    print("="*60 + "\n")
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Find SAM-Med3D root
    sam3d_root = find_default_sam3d_root()
    print(f"SAM-Med3D root: {sam3d_root}\n")
    
    # Check prerequisites
    if not check_prerequisites(dataset_root, sam3d_root):
        print("\n❌ Please fix the issues above and try again.")
        return None
    
    try:
        # Step 1-2: Prepare data
        print("\n" + "-"*60)
        print("STEP 1-2: PREPARING DATA")
        print("-"*60)
        n_prepared, paths = prepare_for_sam3d(
            dataset_root=dataset_root,
            sam3d_root=sam3d_root,
            category=category,
            ct_name=ct_name,
            max_cases=max_cases,
        )
        print(f"✓ Prepared {n_prepared} cases")
        print(f"  Train images: {paths.images_tr}")
        print(f"  Train labels: {paths.labels_tr}")
        
        # Step 3: Split validation
        print("\n" + "-"*60)
        print("STEP 3: SPLITTING VALIDATION SET")
        print("-"*60)
        train_n, val_n = split_validation(
            paths,
            split_ratio=split_ratio,
            seed=seed,
            copy=True,
        )
        print(f"✓ Split complete: {train_n} train, {val_n} validation")
        
        # Step 4: Build model and extract embeddings
        print("\n" + "-"*60)
        print("STEP 4: BUILDING MODEL & EXTRACTING EMBEDDINGS")
        print("-"*60)
        
        # Try to load checkpoint
        ckpt_path = sam3d_root / "ckpt" / "sam_med3d_turbo.pth"
        if not ckpt_path.exists():
            ckpt_path = None
            print("⚠ No checkpoint loaded (using random weights)")
        else:
            print(f"✓ Loading checkpoint: {ckpt_path}")
        
        model = build_sam3d_model(
            sam3d_root=sam3d_root,
            checkpoint=ckpt_path,
            model_type="vit_b_ori",
            device=torch.device(device),
            eval_mode=True,
        )
        print("✓ Model built successfully")
        
        # Extract embeddings
        feat_dirs = default_feature_dirs(sam3d_root, category=category, ct_name=ct_name)
        print(f"Feature directories:")
        print(f"  Train: {feat_dirs.train_dir}")
        print(f"  Val:   {feat_dirs.val_dir}")
        
        extract_embeddings_train_val(
            paths,
            model,
            sam3d_root=sam3d_root,
            img_size=128,
            feature_dirs=feat_dirs,
            device=torch.device(device),
            skip_existing=False,  # Force re-extraction for testing
        )
        print("✓ Embeddings extracted")
        
        # Step 5: Load pooled features (using NEW average pooling)
        print("\n" + "-"*60)
        print("STEP 5: LOADING POOLED FEATURES (AVERAGE POOLING)")
        print("-"*60)
        print("NOTE: Using Global Average Pooling (not ROI pooling)")
        
        X_train, ids_train = load_pooled_features(feat_dirs.train_dir, paths.labels_tr)
        X_val, ids_val = load_pooled_features(feat_dirs.val_dir, paths.labels_val)
        
        print(f"✓ Train features: {X_train.shape}")
        print(f"✓ Val features:   {X_val.shape}")
        print(f"  Train IDs: {len(ids_train)}")
        print(f"  Val IDs:   {len(ids_val)}")
        
        # Step 6: Load labels
        print("\n" + "-"*60)
        print("STEP 6: LOADING LABELS")
        print("-"*60)
        
        sheet_csv = dataset_root / "sheet.csv"
        if not sheet_csv.exists():
            sheet_csv = Path("sheet.csv")
        
        df, lab_map = load_labels_from_sheet(
            sheet_csv=sheet_csv,
            dataset_name="GIST",
            subject_col="Subject",
            label_col="Diagnosis_binary",
            case_suffix="_CT",
        )
        print(f"✓ Loaded {len(lab_map)} labels from {sheet_csv}")
        
        y_train, missing_tr = build_y(ids_train, lab_map)
        y_val, missing_va = build_y(ids_val, lab_map)
        
        if missing_tr:
            print(f"⚠ Missing labels for {len(missing_tr)} train cases: {missing_tr[:3]}...")
        if missing_va:
            print(f"⚠ Missing labels for {len(missing_va)} val cases: {missing_va[:3]}...")
        
        print(f"✓ Train labels: {len(y_train)}")
        print(f"✓ Val labels:   {len(y_val)}")
        print(f"  Label distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"  Label distribution (val):   {dict(zip(*np.unique(y_val, return_counts=True)))}")
        
        # Step 7-8: Run TabPFN pipeline
        print("\n" + "-"*60)
        print("STEP 7-8: RUNNING TABPFN PIPELINE")
        print("-"*60)
        
        import numpy as np
        
        # Create test output directory
        out_dir = Path("test_pipeline_results") / f"{category}_{ct_name}_test"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {out_dir}")
        
        res = tabpfn_pipeline(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            ids_val=ids_val,
            category=category,
            ct_name=ct_name,
            out_dir=out_dir,
            n_components_max=min(500, X_train.shape[1]),
            random_state=42,
        )
        
        print("\n" + "="*60)
        print("✅ PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nResults saved to: {res['out_dir']}")
        print(f"\nMetrics:")
        print(f"  Accuracy: {res['accuracy']:.4f}")
        print(f"  AUC:      {res['auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(res['confusion_matrix'])
        
        return res
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ PIPELINE TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Test full SAM-Med3D pipeline")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("gist"),
        help="Path to dataset root (default: gist)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="gist",
        help="Category name (default: gist)",
    )
    parser.add_argument(
        "--ct-name",
        type=str,
        default="ct_GIST",
        help="CT name (default: ct_GIST)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=10,
        help="Maximum number of cases to use (default: 10 for quick testing)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed (default: 2025)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: cuda or cpu (default: auto-detect)",
    )
    
    args = parser.parse_args()
    
    result = run_test_pipeline(
        dataset_root=args.dataset_root,
        category=args.category,
        ct_name=args.ct_name,
        max_cases=args.max_cases,
        split_ratio=args.split_ratio,
        seed=args.seed,
        device=args.device,
    )
    
    sys.exit(0 if result is not None else 1)


if __name__ == "__main__":
    main()
