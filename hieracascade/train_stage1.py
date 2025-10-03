"""Training script for Stage-1 model"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import argparse
from typing import Dict, List

from hieracascade.models import build_stage1_model
from hieracascade.losses import build_stage1_loss
from hieracascade.dataio import (
    DatasetStage1,
    SiteBalancedSampler,
    load_labels_csv,
    create_site_held_out_splits,
    get_class_weights,
    print_dataset_statistics,
    FINE_TO_IDX,
    COARSE_TO_IDX
)
from hieracascade.metrics import MetricsTracker, format_metrics
from hieracascade.visualization import (
    save_saliency_overlay,
    plot_training_curves,
    plot_confusion_matrix
)


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
    tracker = MetricsTracker()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (volume, y_fine, y_coarse, modality_id, site, study_id) in enumerate(pbar):
        # Move to device
        volume = volume.to(device)
        y_fine = y_fine.to(device)
        modality_id = modality_id.to(device)
        
        # Forward with mixed precision
        optimizer.zero_grad()
        
        with autocast():
            logits, saliency = model(volume, modality_id)
            loss_dict = criterion(logits, saliency, y_fine)
            loss = loss_dict['total']
        
        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Track metrics
        y_pred = torch.argmax(logits, dim=1)
        y_prob = torch.softmax(logits, dim=1)
        
        tracker.update(y_fine, y_pred, y_prob, loss.item())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'ce': loss_dict['ce'].item(),
            'sparsity': loss_dict['sparsity'].item()
        })
    
    metrics = tracker.compute()
    return metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
    save_examples: bool = False,
    output_dir: Path = None
) -> Dict[str, float]:
    """Validate for one epoch"""
    model.eval()
    tracker = MetricsTracker()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    for batch_idx, (volume, y_fine, y_coarse, modality_id, site, study_id) in enumerate(pbar):
        # Move to device
        volume = volume.to(device)
        y_fine = y_fine.to(device)
        modality_id = modality_id.to(device)
        
        # Forward
        logits, saliency = model(volume, modality_id)
        loss_dict = criterion(logits, saliency, y_fine)
        loss = loss_dict['total']
        
        # Track metrics
        y_pred = torch.argmax(logits, dim=1)
        y_prob = torch.softmax(logits, dim=1)
        
        tracker.update(y_fine, y_pred, y_prob, loss.item())
        
        # Save example saliency maps
        if save_examples and batch_idx == 0 and output_dir:
            for i in range(min(2, volume.shape[0])):
                vol = volume[i, 0].cpu().numpy()
                sal = saliency[i, 0].cpu().numpy()
                
                save_path = output_dir / f'saliency_epoch{epoch}_{study_id[i]}.png'
                save_saliency_overlay(vol, sal, str(save_path))
    
    metrics = tracker.compute()
    return metrics


def train_stage1(
    config: Dict,
    data_root: str,
    labels_csv: str,
    output_dir: str,
    fold: int = 0,
    device: str = 'cuda'
):
    """Main training function for Stage-1
    
    Args:
        config: Configuration dictionary
        data_root: Root data directory
        labels_csv: Path to labels CSV
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
    
    # Get class weights
    if config['train'].get('use_class_weights', False):
        class_weights = get_class_weights(train_index, 'y_fine')
        class_weights = torch.tensor(
            [class_weights.get(i, 1.0) for i in range(len(FINE_TO_IDX))],
            dtype=torch.float32
        ).to(device)
    else:
        class_weights = None
    
    # Create datasets
    train_dataset = DatasetStage1(
        train_index,
        data_root=data_root,
        target_spacing=config['geom']['spacing'],
        target_size=tuple(config['geom']['size']),
        augment=config['train']['aug'].get('enabled', True),
        cache_dir=str(output_dir / 'cache')
    )
    
    val_dataset = DatasetStage1(
        val_index,
        data_root=data_root,
        target_spacing=config['geom']['spacing'],
        target_size=tuple(config['geom']['size']),
        augment=False,
        cache_dir=str(output_dir / 'cache')
    )
    
    # Create dataloaders
    batch_size = config['train']['batch_size']
    
    if config['train'].get('site_balanced', True):
        train_sampler = SiteBalancedSampler(train_index, batch_size, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_stage1_model(
        n_classes=len(FINE_TO_IDX),
        backbone=config['model']['name'].replace('stage1_', ''),
        embed_dim=config['model']['embed_dim']
    ).to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Build loss
    criterion = build_stage1_loss(
        n_classes=len(FINE_TO_IDX),
        sparsity_weight=config['train']['lam_sparsity'],
        class_weights=class_weights,
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
        save_examples = (epoch % 5 == 0)
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch,
            save_examples=save_examples,
            output_dir=output_dir / 'visualizations'
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
    parser = argparse.ArgumentParser(description='Train Stage-1 model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--data_root', type=str, required=True, help='Root data directory')
    parser.add_argument('--labels_csv', type=str, required=True, help='Path to labels CSV')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Train
    train_stage1(
        config=config,
        data_root=args.data_root,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        fold=args.fold,
        device=args.device
    )


if __name__ == '__main__':
    main()
