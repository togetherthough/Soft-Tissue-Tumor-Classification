"""Quick start script for HieraCascade using existing sheet.csv

This script demonstrates the complete training pipeline for soft tissue tumor classification:
1. Load labels from the existing sheet.csv file
2. Train Stage-1 model for coarse predictions and saliency
3. Train Stage-2 model for fine-grained hierarchical classification
4. Evaluate results and generate visualizations

The script handles all preprocessing and data loading automatically.
"""

import argparse
from pathlib import Path
import yaml

from hieracascade.dataio import create_index_from_sheet
from hieracascade.train_stage1 import train_stage1
from hieracascade.train_stage2 import train_stage2


def main():
    parser = argparse.ArgumentParser(
        description='Quick start HieraCascade training pipeline for soft tissue tumors'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='data',
        help='Root data directory (default: data)'
    )
    parser.add_argument(
        '--sheet_csv',
        type=str,
        default='data/sheet.csv',
        help='Path to sheet.csv file (default: data/sheet.csv)'
    )
    parser.add_argument(
        '--label_column',
        type=str,
        default='Diagnosis',
        choices=['Diagnosis', 'Diagnosis_binary'],
        help='Column to use for labels (default: Diagnosis). Options: Diagnosis (multi-class), Diagnosis_binary (binary classification)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/hieracascade',
        help='Output directory for results (default: outputs/hieracascade)'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help='Cross-validation fold number (default: 0)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use for training (default: cuda)'
    )
    parser.add_argument(
        '--skip_stage1',
        action='store_true',
        help='Skip Stage-1 training and use existing checkpoint'
    )
    parser.add_argument(
        '--stage1_ckpt',
        type=str,
        help='Path to existing Stage-1 checkpoint (required if skip_stage1)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Step 1: Load labels from the existing sheet.csv file
    print("=" * 60)
    print("STEP 1: Loading labels from sheet.csv")
    print("=" * 60)
    
    labels_csv = output_dir / 'labels.csv'
    index = create_index_from_sheet(
        data_root=args.data_root,
        sheet_path=args.sheet_csv,
        label_column=args.label_column,
        output_csv=str(labels_csv)
    )
    
    print(f"\n✓ Created labels CSV with {len(index)} studies")
    print(f"  Saved to: {labels_csv}")
    
    # Step 2: Train Stage-1 model (or skip if checkpoint is provided)
    stage1_output = output_dir / f'stage1/fold{args.fold}'
    
    if args.skip_stage1:
        if not args.stage1_ckpt:
            raise ValueError("--stage1_ckpt is required when --skip_stage1 is set")
        stage1_checkpoint = args.stage1_ckpt
        print(f"\n⏩ Skipping Stage-1 training, using existing checkpoint: {stage1_checkpoint}")
    else:
        print("\n" + "=" * 60)
        print("STEP 2: Training Stage-1 (Coarse + Saliency)")
        print("=" * 60)
        
        # Load Stage-1 configuration
        config_path = Path(__file__).parent / 'configs' / 'stage1.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Train Stage-1 model
        train_stage1(
            config=config,
            data_root=args.data_root,
            labels_csv=str(labels_csv),
            output_dir=str(stage1_output),
            fold=args.fold,
            device=args.device
        )
        
        stage1_checkpoint = str(stage1_output / 'checkpoint_best.pt')
        print(f"\n✓ Stage-1 training complete!")
        print(f"  Best checkpoint saved to: {stage1_checkpoint}")
    
    # Step 3: Train Stage-2 model
    print("\n" + "=" * 60)
    print("STEP 3: Training Stage-2 (Hierarchical Classification)")
    print("=" * 60)
    
    stage2_output = output_dir / f'stage2/fold{args.fold}'
    
    # Load Stage-2 configuration
    config_path = Path(__file__).parent / 'configs' / 'stage2.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Train Stage-2 model
    train_stage2(
        config=config,
        data_root=args.data_root,
        labels_csv=str(labels_csv),
        stage1_checkpoint=stage1_checkpoint,
        output_dir=str(stage2_output),
        fold=args.fold,
        device=args.device
    )
    
    print(f"\n✓ Stage-2 training complete!")
    print(f"  Best checkpoint saved to: {stage2_output / 'checkpoint_best.pt'}")
    
    # Summary and next steps
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  1. Evaluate the model: python -m hieracascade.evaluate \\")
    print(f"       --checkpoint {stage2_output / 'checkpoint_best.pt'} \\")
    print(f"       --stage stage2 \\")
    print(f"       --data_root {args.data_root} \\")
    print(f"       --labels_csv {labels_csv} \\")
    print(f"       --stage1_ckpt {stage1_checkpoint} \\")
    print(f"       --output_dir {stage2_output / 'eval'} \\")
    print(f"       --fold {args.fold}")
    print("\n  2. View training results:")
    print(f"       - Training curves: {stage2_output / 'plots' / 'training_curves.png'}")
    print(f"       - Saliency maps: {stage1_output / 'visualizations'}")


if __name__ == '__main__':
    main()
