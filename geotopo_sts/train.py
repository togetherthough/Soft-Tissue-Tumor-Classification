"""
Training script for GeoTopo-STS
Includes loss functions, metrics, and training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from tqdm import tqdm
import json

from .models import GeoTopoSTS
from .models.fusion_head import hierarchical_loss, compute_class_weights
from .dataio import GeoTopoDataset, get_train_transforms, get_val_transforms


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        preds: (N,) predicted class indices
        targets: (N,) true class indices
        num_classes: total number of classes
    
    Returns:
        dict with metrics
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        f1_score, precision_score, recall_score,
        confusion_matrix
    )
    
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(targets, preds)
    
    # Balanced accuracy
    metrics['balanced_accuracy'] = balanced_accuracy_score(targets, preds)
    
    # Macro F1 (main metric)
    metrics['macro_f1'] = f1_score(targets, preds, average='macro', zero_division=0)
    
    # Weighted F1
    metrics['weighted_f1'] = f1_score(targets, preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    per_class_recall = recall_score(targets, preds, average=None, zero_division=0, labels=range(num_classes))
    metrics['mean_recall'] = per_class_recall.mean()
    
    # Confusion matrix
    cm = confusion_matrix(targets, preds, labels=range(num_classes))
    metrics['confusion_matrix'] = cm
    
    return metrics


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error.
    
    Args:
        probs: (N, K) predicted probabilities
        targets: (N,) true class indices
        n_bins: number of bins
    
    Returns:
        ECE score
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(float)
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


class Trainer:
    """Training manager for GeoTopo-STS"""
    
    def __init__(
        self,
        model: GeoTopoSTS,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda',
        output_dir: str = './outputs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training config
        train_cfg = config.get('training', {})
        self.epochs = train_cfg.get('epochs', 200)
        self.grad_accum = train_cfg.get('gradient_accumulation', 4)
        self.use_amp = train_cfg.get('mixed_precision', True)
        
        # Optimizer
        self._setup_optimizer(train_cfg.get('optimizer', {}))
        
        # Scheduler
        self._setup_scheduler(train_cfg.get('scheduler', {}))
        
        # Loss config
        loss_cfg = train_cfg.get('loss', {})
        self.ce_weight = loss_cfg.get('ce_weight', 1.0)
        self.hier_weight = loss_cfg.get('hierarchy_weight', 0.3)
        self.use_hierarchy = config.get('head', {}).get('hierarchy', {}).get('enabled', True)
        
        # Class weights for imbalanced data
        self.class_weights = None  # Set externally if needed
        
        # Hierarchy ancestors (set externally if using hierarchical loss)
        self.ancestors = None
        
        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.best_metric = 0.0
        self.current_epoch = 0
        self.history = {'train': [], 'val': []}
    
    def _setup_optimizer(self, opt_cfg: Dict):
        """Setup optimizer with parameter groups"""
        lr_voxel = opt_cfg.get('lr_voxel', 3e-4)
        lr_graph = opt_cfg.get('lr_graph', 1e-3)
        lr_mlp = opt_cfg.get('lr_mlp', 1e-3)
        weight_decay = opt_cfg.get('weight_decay', 0.05)
        
        # Parameter groups
        voxel_params = list(self.model.voxel_encoder.parameters())
        graph_params = []
        mlp_params = []
        
        if hasattr(self.model, 'mesh_encoder'):
            graph_params.extend(self.model.mesh_encoder.parameters())
        if hasattr(self.model, 'topo_encoder'):
            mlp_params.extend(self.model.topo_encoder.parameters())
        
        mlp_params.extend(self.model.fusion.parameters())
        mlp_params.extend(self.model.head.parameters())
        
        param_groups = [
            {'params': voxel_params, 'lr': lr_voxel},
            {'params': graph_params, 'lr': lr_graph},
            {'params': mlp_params, 'lr': lr_mlp}
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay
        )
    
    def _setup_scheduler(self, sched_cfg: Dict):
        """Setup learning rate scheduler"""
        sched_name = sched_cfg.get('name', 'onecycle')
        
        if sched_name == 'onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[g['lr'] for g in self.optimizer.param_groups],
                epochs=self.epochs,
                steps_per_epoch=len(self.train_loader) // self.grad_accum,
                pct_start=sched_cfg.get('pct_start', 0.2)
            )
        elif sched_name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=1e-6
            )
        else:
            self.scheduler = None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        loss_meter = AverageMeter()
        ce_meter = AverageMeter()
        hier_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}/{self.epochs}')
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = self._to_device(batch)
            
            # Forward
            with autocast(enabled=self.use_amp):
                logits = self.model(batch)
                targets = batch['label']
                
                # Cross-entropy loss
                if self.class_weights is not None:
                    ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
                else:
                    ce_loss = F.cross_entropy(logits, targets)
                
                # Hierarchical loss
                if self.use_hierarchy and self.ancestors is not None:
                    hier_loss = hierarchical_loss(
                        logits, targets, self.ancestors,
                        lambda_h=self.hier_weight
                    )
                else:
                    hier_loss = torch.tensor(0.0, device=self.device)
                
                # Total loss
                loss = self.ce_weight * ce_loss + self.hier_weight * hier_loss
                loss = loss / self.grad_accum
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.grad_accum == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if self.scheduler is not None and isinstance(
                    self.scheduler, torch.optim.lr_scheduler.OneCycleLR
                ):
                    self.scheduler.step()
            
            # Update meters
            loss_meter.update(loss.item() * self.grad_accum, targets.size(0))
            ce_meter.update(ce_loss.item(), targets.size(0))
            hier_meter.update(hier_loss.item(), targets.size(0))
            
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'ce': f'{ce_meter.avg:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return {
            'loss': loss_meter.avg,
            'ce_loss': ce_meter.avg,
            'hier_loss': hier_meter.avg
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        loss_meter = AverageMeter()
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            batch = self._to_device(batch)
            
            logits = self.model(batch)
            targets = batch['label']
            
            # Loss
            loss = F.cross_entropy(logits, targets)
            loss_meter.update(loss.item(), targets.size(0))
            
            # Predictions
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        # Concatenate
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)
        
        # Compute metrics
        num_classes = self.config['head']['n_classes']
        metrics = compute_metrics(all_preds, all_targets, num_classes)
        metrics['loss'] = loss_meter.avg
        metrics['ece'] = compute_ece(all_probs, all_targets)
        
        return metrics
    
    def train(self):
        """Full training loop"""
        print(f"Starting training for {self.epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch()
            self.history['train'].append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.history['val'].append(val_metrics)
            
            # Scheduler step (for non-OneCycle)
            if self.scheduler is not None and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                self.scheduler.step()
            
            # Log
            print(f"\nEpoch {self.current_epoch}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Macro-F1: {val_metrics['macro_f1']:.4f} | "
                  f"Bal-Acc: {val_metrics['balanced_accuracy']:.4f} | "
                  f"ECE: {val_metrics['ece']:.4f}")
            
            # Save best model
            metric_to_monitor = val_metrics['macro_f1']
            if metric_to_monitor > self.best_metric:
                self.best_metric = metric_to_monitor
                self.save_checkpoint('best_model.pth')
                print(f"✓ Saved best model (Macro-F1: {self.best_metric:.4f})")
            
            # Save checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch}.pth')
            
            # Save history
            self.save_history()
        
        print(f"\nTraining complete! Best Macro-F1: {self.best_metric:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def save_history(self):
        """Save training history"""
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {
            'train': self.history['train'],
            'val': [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in epoch.items()} for epoch in self.history['val']]
        }
        
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(history_serializable, f, indent=2)
    
    def _to_device(self, batch: Dict) -> Dict:
        """Move batch to device"""
        for key in ['volume', 'mask', 'rim', 'label', 'ph_features']:
            if key in batch:
                batch[key] = batch[key].to(self.device)
        
        # Handle mesh data
        if 'mesh_data' in batch and batch['mesh_data'] is not None:
            if isinstance(batch['mesh_data'], dict):
                for key in ['vertices', 'features', 'edge_index']:
                    if key in batch['mesh_data']:
                        batch['mesh_data'][key] = batch['mesh_data'][key].to(self.device)
        
        return batch


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GeoTopo-STS')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--data', type=str, required=True, help='Path to preprocessed data')
    parser.add_argument('--output', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create datasets
    train_dataset = GeoTopoDataset(
        args.data,
        split='train',
        transform=get_train_transforms(config.get('augmentation')),
        config=config
    )
    
    val_dataset = GeoTopoDataset(
        args.data,
        split='val',
        transform=get_val_transforms(),
        config=config
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    # Create model
    model = GeoTopoSTS(config['model'])
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
        output_dir=args.output
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
