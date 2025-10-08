#!/usr/bin/env python
"""
Minimal example: Test SAM-Med3D feature quality for GIST dataset

This is the simplest possible usage - just 3 function calls.
"""

from pathlib import Path
from med3pipe.data.prepare import prepare_for_sam3d, split_validation
from med3pipe.sam.core import load_labels_from_sheet
from med3pipe.training import run_classification_head_experiment

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "gist"
SHEET_CSV = DATASET_ROOT / "sheet.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "feature_test_minimal"

print("=" * 60)
print("Minimal SAM-Med3D Feature Quality Test")
print("=" * 60)

# Step 1: Prepare dataset
print("\n[1/3] Preparing dataset...")
prepared, paths = prepare_for_sam3d(
    dataset_root=DATASET_ROOT,
    category="gist",
    ct_name="ct_GIST",
)
split_validation(paths, split_ratio=0.8, seed=2025, copy=True)
print(f"✓ Prepared {prepared} cases")

# Step 2: Load labels
print("\n[2/3] Loading labels...")
df, lab_map = load_labels_from_sheet(
    sheet_csv=SHEET_CSV,
    dataset_name="GIST",
    subject_col="Subject",
    label_col="Diagnosis_binary",
    case_suffix="_CT",
)
print(f"✓ Loaded {len(lab_map)} labels")
print(f"  Benign: {(df['label'] == 0).sum()}, Malignant: {(df['label'] == 1).sum()}")

# Step 3: Test features
print("\n[3/3] Testing SAM-Med3D features...")
results = run_classification_head_experiment(
    paths=paths,
    lab_map=lab_map,
    freeze_encoder=True,  # Test pretrained features
    num_epochs=10,
    batch_size=4,
    learning_rate=1e-3,
    output_dir=OUTPUT_DIR,
)

# Results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Best Validation AUC: {results['best_auc']:.4f}")
print(f"Final Accuracy:      {results['final_metrics'].accuracy:.4f}")

if results['best_auc'] > 0.7:
    print("\n✅ GOOD: Features are discriminative!")
    print("   → Proceed with TabPFN/LoCalPFN pipeline")
elif results['best_auc'] > 0.6:
    print("\n⚠️  MODERATE: Features have some signal")
    print("   → Try fine-tuning or check data quality")
else:
    print("\n❌ POOR: Features are not discriminative")
    print("   → Check data quality or try different approach")

print(f"\nDetailed results saved to: {OUTPUT_DIR}")
print("=" * 60)
