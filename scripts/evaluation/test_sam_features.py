#!/usr/bin/env python
"""
Script to test SAM-Med3D feature quality by training a classification head.

This script helps evaluate whether SAM-Med3D features are discriminative for
tumor classification before using them in the full TabPFN/LoCalPFN pipeline.

Usage:
    python scripts/test_sam_features.py --dataset gist --epochs 10 --freeze
    python scripts/test_sam_features.py --dataset lipo --epochs 20 --no-freeze
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from med3pipe.data.prepare import prepare_for_sam3d, split_validation, find_default_sam3d_root
from med3pipe.sam.core import load_labels_from_sheet
from med3pipe.training.classification_head import run_classification_head_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Test SAM-Med3D feature quality with classification head"
    )
    
    # Dataset args
    parser.add_argument(
        "--dataset",
        type=str,
        default="gist",
        help="Dataset name (e.g., 'gist', 'lipo')",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to dataset root (default: data/<dataset>)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Category name (default: same as dataset)",
    )
    parser.add_argument(
        "--ct-name",
        type=str,
        default=None,
        help="CT name (default: ct_<DATASET>)",
    )
    
    # Label args
    parser.add_argument(
        "--sheet-csv",
        type=Path,
        default=None,
        help="Path to sheet.csv with labels",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name in sheet.csv (e.g., 'GIST')",
    )
    parser.add_argument(
        "--case-suffix",
        type=str,
        default="_CT",
        help="Case suffix for label matching (default: _CT)",
    )
    
    # Model args
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to SAM-Med3D checkpoint",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_b_ori",
        help="SAM-Med3D model type",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="Input image size (default: 128)",
    )
    
    # Training args
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Freeze encoder (train only classification head)",
    )
    parser.add_argument(
        "--no-freeze",
        dest="freeze",
        action="store_false",
        help="Don't freeze encoder (fine-tune end-to-end)",
    )
    parser.set_defaults(freeze=True)
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate (default: 0.3)",
    )
    
    # Split args
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
    
    # Output args
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/classification_head/<dataset>)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device ('cuda' or 'cpu', default: auto)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers (default: 2)",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    sam3d_root = find_default_sam3d_root()
    
    if args.dataset_root is None:
        # Try data/<dataset> first, then <dataset>
        candidates = [
            PROJECT_ROOT / "data" / args.dataset,
            PROJECT_ROOT / args.dataset,
        ]
        args.dataset_root = next((c for c in candidates if c.exists()), candidates[0])
    
    if args.category is None:
        args.category = args.dataset
    
    if args.ct_name is None:
        args.ct_name = f"ct_{args.dataset.upper()}"
    
    if args.dataset_name is None:
        args.dataset_name = args.dataset.upper()
    
    if args.sheet_csv is None:
        # Try dataset_root/sheet.csv first
        if (args.dataset_root / "sheet.csv").exists():
            args.sheet_csv = args.dataset_root / "sheet.csv"
        else:
            args.sheet_csv = PROJECT_ROOT / "sheet.csv"
    
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "results" / "classification_head" / args.dataset
    
    print("\n" + "="*60)
    print("Configuration")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Category: {args.category}")
    print(f"CT name: {args.ct_name}")
    print(f"Sheet CSV: {args.sheet_csv}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Freeze encoder: {args.freeze}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {args.output_dir}")
    print("="*60 + "\n")
    
    # Step 1: Prepare dataset
    print("[Step 1/4] Preparing dataset for SAM-Med3D...")
    prepared, paths = prepare_for_sam3d(
        dataset_root=args.dataset_root,
        sam3d_root=sam3d_root,
        category=args.category,
        ct_name=args.ct_name,
        case_glob=None,
        max_cases=None,
    )
    print(f"✓ Prepared {prepared} cases")
    
    # Step 2: Split validation
    print("\n[Step 2/4] Creating validation split...")
    split_validation(
        paths,
        split_ratio=args.split_ratio,
        seed=args.seed,
        copy=True,
    )
    print("✓ Validation split created")
    
    # Step 3: Load labels
    print("\n[Step 3/4] Loading labels...")
    df, lab_map = load_labels_from_sheet(
        sheet_csv=args.sheet_csv,
        dataset_name=args.dataset_name,
        subject_col="Subject",
        label_col="Diagnosis_binary",
        case_suffix=args.case_suffix,
    )
    print(f"✓ Loaded {len(lab_map)} labels")
    print(f"  Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Step 4: Run classification head experiment
    print("\n[Step 4/4] Running classification head experiment...")
    results = run_classification_head_experiment(
        paths=paths,
        lab_map=lab_map,
        sam3d_root=sam3d_root,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        img_size=args.img_size,
        device=args.device,
        freeze_encoder=args.freeze,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)
    print(f"Best AUC: {results['best_auc']:.4f}")
    print(f"Final Accuracy: {results['final_metrics'].accuracy:.4f}")
    print(f"Results saved to: {args.output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
