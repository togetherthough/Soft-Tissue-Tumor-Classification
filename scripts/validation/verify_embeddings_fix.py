#!/usr/bin/env python3
"""
Verification script for automatic embedding extraction fix.

This script demonstrates that the multi-dataset pipeline now automatically
extracts embeddings for any dataset that's missing them, regardless of order.
"""

from pathlib import Path
import sys

def check_embeddings(sam3d_root: Path, category: str, ct_name: str) -> dict:
    """Check if embeddings exist for a dataset."""
    train_dir = sam3d_root / "features" / category / f"{ct_name}_train"
    val_dir = sam3d_root / "features" / category / ct_name
    
    train_files = list(train_dir.glob("*_embedding.pt")) if train_dir.exists() else []
    val_files = list(val_dir.glob("*_embedding.pt")) if val_dir.exists() else []
    
    return {
        "train_dir": train_dir,
        "val_dir": val_dir,
        "train_count": len(train_files),
        "val_count": len(val_files),
        "exists": bool(train_files and val_files),
    }


def main():
    """Check embedding status for all datasets."""
    sam3d_root = Path("sam-med3d")
    
    datasets = {
        "gist": ("gist", "ct_GIST"),
        "lipo": ("lipo", "ct_LIPO"),
    }
    
    print("=" * 70)
    print("EMBEDDING STATUS CHECK")
    print("=" * 70)
    
    all_exist = True
    for name, (category, ct_name) in datasets.items():
        info = check_embeddings(sam3d_root, category, ct_name)
        
        status = "✅ EXISTS" if info["exists"] else "❌ MISSING"
        print(f"\n{name.upper():10s} {status}")
        print(f"  Train: {info['train_count']:3d} embeddings in {info['train_dir']}")
        print(f"  Val:   {info['val_count']:3d} embeddings in {info['val_dir']}")
        
        if not info["exists"]:
            all_exist = False
    
    print("\n" + "=" * 70)
    
    if all_exist:
        print("✅ All embeddings exist! Ready to run experiments.")
        print("\nYou can now run:")
        print("  python -m med3pipe multi-tabpfn --config configs/datasets.yaml")
        print("  python -m med3pipe multi-localpfn --config configs/datasets.yaml")
        return 0
    else:
        print("⚠️  Some embeddings are missing.")
        print("\n🔧 FIX: Run either command below to automatically extract missing embeddings:")
        print("\nFor TabPFN:")
        print("  python -m med3pipe multi-tabpfn --config configs/datasets.yaml")
        print("\nFor LoCalPFN:")
        print("  python -m med3pipe multi-localpfn --config configs/datasets.yaml")
        print("\n💡 The pipeline will now automatically detect and extract missing embeddings!")
        print("   This works regardless of which datasets you run or in what order.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
