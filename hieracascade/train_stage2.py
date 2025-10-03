"""Training script for Stage-2 model"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import argparse
from typing import Dict

from hieracascade.models import build_stage1_model, build_stage2_model
from hieracascade.losses import build_stage2_loss
from hieracascade.dataio import (
    DatasetStage1,
    DatasetStage2,
    load_labels_csv,
    create_site_held_out_splits,
    get_class_weights,
    print_dataset_statistics,
    FINE_TO_IDX,
    COARSE_TO_IDX,
    CLASS_HIERARCHY
)
from hieracascade.metrics import MetricsTracker, format_metrics
from hieracascade.visualization import plot_training_curves, visualize_crop_locations


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: GradScaler,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    tracker_fine = MetricsTracker()
    tracker_coarse = MetricsTracker()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (crops, y_fine, y_coarse, modality_id, site, study_id) in enumerate(pbar):
        # Move to device
        crops = crops.to(device)
        y_fine = y_fine.to(device)
        y_coarse = y_coarse.to(device)
        modality_id = modality_id.to(device)
        
        # Forward with mixed precision
        optimizer.zero_grad()
        
        with autocast():
            logits_fine, logits_coarse, _ = model(crops, modality_id)
            loss_dict = criterion(logits_fine, logits_coarse, y_fine, y_coarse)
            loss = loss_dict['total']
        
        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Track metrics
        y_pred_fine = torch.argmax(logits_fine, dim=1)
        y_pred_coarse = torch.argmax(logits_coarse, dim=1)
        y_prob_fine = torch.softmax(logits_fine, dim=1)
        y_prob_coarse = torch.softmax(logits_coarse, dim=1)
        
        tracker_fine.update(y_fine, y_pred_fine, y_prob_fine, loss.item())
        tracker_coarse.update(y_coarse, y_pred_coarse, y_prob_coarse)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'L_f': loss_dict['fine'].item(),
            'L_c': loss_dict['coarse'].item(),
            'L_cons': loss_dict['consistency'].item()
        })
    
    metrics = tracker_fine.compute()
    metrics.update({f'coarse_{k}': v for k, v in tracker_coarse.compute().items() if k != 'loss'})
    
    return metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Validate for one epoch"""
    model.eval()
    tracker_fine = MetricsTracker()
    tracker_coarse = MetricsTracker()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    for batch_idx, (crops, y_fine, y_coarse, modality_id, site, study_id) in enumerate(pbar):
        # Move to device
        crops = crops.to(device)
        y_fine = y_fine.to(device)
        y_coarse = y_coarse.to(device)
        modality_id = modality_id.to(device)
        
        # Forward
        logits_fine, logits_coarse, _ = model(crops, modality_id)
        loss_dict = criterion(logits_fine, logits_coarse, y_fine, y_coarse)
        loss = loss_dict['total']
        
        # Track metrics
        y_pred_fine = torch.argmax(logits_fine, dim=1)
        y_pred_coarse = torch.argmax(logits_coarse, dim=1)
        y_prob_fine = torch.softmax(logits_fine, dim=1)
        y_prob_coarse = torch.softmax(logits_coarse, dim=1)
        
        tracker_fine.update(y_fine, y_pred_fine, y_prob_fine, loss.item())
        tracker_coarse.update(y_coarse, y_pred_coarse, y_prob_coarse)
    
    metrics = tracker_fine.compute()
    metrics.update({f'coarse_{k}': v for k, v in tracker_coarse.compute().items() if k != 'loss'})
    
    return metrics


def train_stage2(
    config: Dict,
    data_root: str,
    labels_csv: str,
    stage1_checkpoint: str,
    output_dir: str,
    fold: int = 0,
    device: str = 'cuda'
):
    """Main training function for Stage-2
    
    Args:
        config: Configuration dictionary
        data_root: Root data directory
        labels_csv: Path to labels CSV
        stage1_checkpoint: Path to trained Stage-1 checkpoint
        output_dir: Output directory for checkpoints and logs
        fold: Fold number for cross-validation
        device: Device to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Load data index
    print("Loading data index...")
    index = load_labels_csv(labels_csv)
    print_dataset_statistics(index)
    
    # Create splits
    print(f"\nCreating site-held-out splits (fold {fold})...")
    splits = create_site_held_out_splits(index)
    train_index, val_index = splits[fold]
    
    print(f"Train: {len(train_index)} samples")
    print(f"Val: {len(val_index)} samples")
    
    # Load Stage-1 model
    print("\nLoading Stage-1 model...")
    stage1_ckpt = torch.load(stage1_checkpoint, map_location=device)
    
    stage1_model = build_stage1_model(
        n_classes=len(FINE_TO_IDX),
        backbone=stage1_ckpt['config']['model']['name'].replace('stage1_', ''),
        embed_dim=stage1_ckpt['config']['model']['embed_dim']
    ).to(device)
    
    stage1_model.load_state_dict(stage1_ckpt['model_state_dict'])
    stage1_model.eval()
    
    # Freeze Stage-1
    for param in stage1_model.parameters():
        param.requires_grad = False
    
    # Create Stage-1 datasets (for crop generation)
    stage1_train_dataset = DatasetStage1(
        train_index,
        data_root=data_root,
        target_spacing=config['geom']['spacing'],
        target_size=tuple(config['geom']['size_stage1']),
        augment=False,
        cache_dir=str(output_dir / 'cache')
    )
    
    stage1_val_dataset = DatasetStage1(
        val_index,
        data_root=data_root,
        target_spacing=config['geom']['spacing'],
        target_size=tuple(config['geom']['size_stage1']),
        augment=False,
        cache_dir=str(output_dir / 'cache')
    )
    
    # Create Stage-2 datasets
    print("\nCreating Stage-2 datasets with crop proposals...")
    train_dataset = DatasetStage2(
        stage1_train_dataset,
        stage1_model,
        K=config['proposals']['K'],
        crop_size=config['proposals']['crop_size'],
        nms_distance=config['proposals']['nms_dist'],
        top_p=config['proposals']['top_p'],
        cache_centers=True,
        device=device
    )
    
    val_dataset = DatasetStage2(
        stage1_val_dataset,
        stage1_model,
        K=config['proposals']['K'],
        crop_size=config['proposals']['crop_size'],
        nms_distance=config['proposals']['nms_dist'],
        top_p=config['proposals']['top_p'],
        cache_centers=True,
        device=device
    )
    
    # Create dataloaders
    batch_size = config['train']['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Reduced workers since Stage-1 inference is on GPU
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Get class weights
    if config['train'].get('use_class_weights', False):
        class_weights_fine = get_class_weights(train_index, 'y_fine')
        class_weights_fine = torch.tensor(
            [class_weights_fine.get(i, 1.0) for i in range(len(FINE_TO_IDX))],
            dtype=torch.float32
        ).to(device)
        
        class_weights_coarse = get_class_weights(train_index, 'y_coarse')
        class_weights_coarse = torch.tensor(
            [class_weights_coarse.get(i, 1.0) for i in range(len(COARSE_TO_IDX))],
            dtype=torch.float32
        ).to(device)
    else:
        class_weights_fine = None
        class_weights_coarse = None
    
    # Build Stage-2 model
    print("\nBuilding Stage-2 model...")
    model = build_stage2_model(
        n_fine=len(FINE_TO_IDX),
        n_coarse=len(COARSE_TO_IDX),
        backbone=config['model']['name'].replace('stage2_', ''),
        embed_dim=config['model']['emb_dim'],
        pooling=config['model']['pool']
    ).to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Build fine-to-coarse mapping
    fine_to_coarse_map = {}
    for fine_name, fine_idx in FINE_TO_IDX.items():
        coarse_name = CLASS_HIERARCHY.get(fine_name, 'other')
        coarse_idx = COARSE_TO_IDX[coarse_name]
        fine_to_coarse_map[fine_idx] = coarse_idx
    
    # Build loss
    criterion = build_stage2_loss(
        n_fine=len(FINE_TO_IDX),
        n_coarse=len(COARSE_TO_IDX),
        fine_to_coarse_map=fine_to_coarse_map,
        alpha=config['train']['alpha'],
        beta=config['train']['beta'],
        class_weights_fine=class_weights_fine,
        class_weights_coarse=class_weights_coarse,
        focal_gamma=config['train'].get('focal_gamma', 0.0)
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )
    
    # Learning rate scheduler
    n_epochs = config['train']['epochs']
    warmup_epochs = config['train']['warmup_epochs']
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs - warmup_epochs,
        eta_min=config['train']['lr'] * 0.01
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    print("\nStarting training...")
    best_f1 = 0.0
    train_history = []
    val_history = []
    
    for epoch in range(1, n_epochs + 1):
        # Warmup learning rate
        if epoch <= warmup_epochs:
            lr = config['train']['lr'] * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        train_history.append(train_metrics)
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        val_history.append(val_metrics)
        
        # Step scheduler (after warmup)
        if epoch > warmup_epochs:
            scheduler.step()
        
        # Print metrics
        print(f"\nEpoch {epoch}/{n_epochs}")
        print("Train:", format_metrics(train_metrics, prefix='  '))
        print("Val:", format_metrics(val_metrics, prefix='  '))
        
        # Save checkpoint
        current_f1 = val_metrics.get('macro_f1', 0.0)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': val_metrics,
            'config': config
        }
        
        # Save latest
        torch.save(checkpoint, output_dir / 'checkpoint_latest.pt')
        
        # Save best
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(checkpoint, output_dir / 'checkpoint_best.pt')
            print(f"  → New best F1: {best_f1:.4f}")
    
    # Plot training curves
    (output_dir / 'plots').mkdir(exist_ok=True)
    plot_training_curves(
        train_history,
        val_history,
        output_path=str(output_dir / 'plots' / 'training_curves.png')
    )
    
    print(f"\nTraining complete! Best val F1: {best_f1:.4f}")
    print(f"Outputs saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Stage-2 model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--data_root', type=str, required=True, help='Root data directory')
    parser.add_argument('--labels_csv', type=str, required=True, help='Path to labels CSV')
    parser.add_argument('--stage1_ckpt', type=str, required=True, help='Stage-1 checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Train
    train_stage2(
        config=config,
        data_root=args.data_root,
        labels_csv=args.labels_csv,
        stage1_checkpoint=args.stage1_ckpt,
        output_dir=args.output_dir,
        fold=args.fold,
        device=args.device
    )


if __name__ == '__main__':
    main()
