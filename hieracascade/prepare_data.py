"""Prepare dataset: scan data directory and create labels CSV

This utility scans the data directory structure and generates a CSV file
with all the necessary labels and metadata for training.
"""

import argparse
from pathlib import Path

from hieracascade.dataio import (
    scan_data_directory,
    save_labels_csv,
    print_dataset_statistics
)


def main():
    parser = argparse.ArgumentParser(
        description='Scan data directory and create labels CSV for HieraCascade'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root data directory containing category subdirectories'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='labels.csv',
        help='Output CSV file path (default: labels.csv)'
    )
    
    args = parser.parse_args()
    
    # Scan the data directory
    print(f"Scanning data directory: {args.data_root}")
    index = scan_data_directory(args.data_root)
    
    if len(index) == 0:
        print("ERROR: No valid studies found in data directory!")
        print("\nExpected directory structure:")
        print("  data_root/")
        print("    <category>/")
        print("      <study_id>_<modality>/")
        print("        1/")
        print("          NIFTI/")
        print("            image.nii.gz")
        return
    
    # Print statistics
    print_dataset_statistics(index)
    
    # Save the CSV file
    save_labels_csv(index, args.output_csv)
    
    print(f"\n✓ Successfully created labels CSV with {len(index)} studies")
    print(f"  Output: {args.output_csv}")
    print("\nNext steps:")
    print(f"  1. Train Stage-1: python -m hieracascade.train_stage1 --config hieracascade/configs/stage1.yaml --data_root {args.data_root} --labels_csv {args.output_csv} --output_dir outputs/stage1/fold0")
    print(f"  2. Train Stage-2: python -m hieracascade.train_stage2 --config hieracascade/configs/stage2.yaml --data_root {args.data_root} --labels_csv {args.output_csv} --stage1_ckpt outputs/stage1/fold0/checkpoint_best.pt --output_dir outputs/stage2/fold0")


if __name__ == '__main__':
    main()
