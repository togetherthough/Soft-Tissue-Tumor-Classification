from __future__ import annotations

"""
med3pipe.training.classification_head

Adds a simple classification head on top of SAM-Med3D 3D encodings to evaluate
feature quality for tumor classification (benign vs malignant).

This is a diagnostic tool to check if the SAM-Med3D features are discriminative
before using them in the full TabPFN/LoCalPFN pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import SimpleITK as sitk

from ..sam.core import (
    build_sam3d_model,
    make_pre_transform,
    load_volume_tensor,
)
from ..data.prepare import Sam3DPaths, find_default_sam3d_root
from ..tabular.lesion_filter import LesionSizeFilter


class TumorClassificationHead(nn.Module):
    """Simple classification head for SAM-Med3D features.
    
    Takes the 3D feature map from SAM-Med3D image encoder and applies:
    1. Global Average Pooling (GAP)
    2. Dropout
    3. Linear classifier
    """
    
    def __init__(
        self,
        in_channels: int = 768,  # ViT-B output channels
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W) feature map from image encoder
        Returns:
            (B, num_classes) logits
        """
        # Global average pooling: (B, C, D, H, W) -> (B, C, 1, 1, 1)
        x = self.gap(x)
        # Flatten: (B, C, 1, 1, 1) -> (B, C)
        x = x.view(x.size(0), -1)
        # Dropout and classification
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SAMWithClassificationHead(nn.Module):
    """Wrapper that combines SAM-Med3D encoder with classification head."""
    
    def __init__(
        self,
        sam_model: nn.Module,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        
        # Infer encoder output channels
        with torch.no_grad():
            dummy = torch.randn(1, 1, 128, 128, 128)
            if next(self.image_encoder.parameters()).is_cuda:
                dummy = dummy.cuda()
            enc_out = self.image_encoder(dummy)
            in_channels = enc_out.size(1)
        
        self.classifier = TumorClassificationHead(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        if freeze_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            print(f"[INFO] SAM-Med3D encoder frozen. Training only classification head.")
        else:
            print(f"[INFO] SAM-Med3D encoder unfrozen. Fine-tuning end-to-end.")
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, D, H, W) input volume
        Returns:
            (B, num_classes) logits
        """
        features = self.image_encoder(x)
        logits = self.classifier(features)
        return logits


class TumorDataset(Dataset):
    """Dataset for tumor classification from prepared SAM-Med3D data."""
    
    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        pre_transform=None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.pre_transform = pre_transform
        assert len(image_paths) == len(labels)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load volume
        vol = load_volume_tensor(img_path, pre_transform=self.pre_transform)
        vol = vol.squeeze(0)  # Remove batch dim: (1, D, H, W)
        
        return vol, label


def prepare_dataloaders(
    paths: Sam3DPaths,
    lab_map: dict,
    batch_size: int = 4,
    num_workers: int = 2,
    img_size: int = 128,
    lesion_filter: Optional[LesionSizeFilter] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Prepare train and val dataloaders from prepared SAM-Med3D paths."""
    pre_transform = make_pre_transform(img_size=img_size)
    
    # Get valid cases if filtering is enabled
    valid_cases = None
    if lesion_filter is not None and lesion_filter.is_enabled():
        lesion_filter.print_filter_summary()
        valid_cases = lesion_filter.get_valid_cases()
    
    # Helper to extract case ID from .nii.gz filename
    def get_case_id(img_path: Path) -> str:
        """Extract case ID from filename, handling .nii.gz extension."""
        name = img_path.name
        if name.endswith('.nii.gz'):
            return name[:-7]  # Remove .nii.gz
        elif name.endswith('.nii'):
            return name[:-4]  # Remove .nii
        else:
            return img_path.stem
    
    # Training data
    train_imgs = sorted(paths.images_tr.glob("*.nii.gz"))
    train_labels = []
    train_valid = []
    for img_path in train_imgs:
        case_id = get_case_id(img_path)
        if case_id in lab_map:
            # Apply lesion size filter if enabled
            if valid_cases is not None and case_id not in valid_cases:
                continue
            train_labels.append(lab_map[case_id])
            train_valid.append(img_path)
    
    # Validation data
    val_imgs = sorted(paths.images_val.glob("*.nii.gz"))
    val_labels = []
    val_valid = []
    for img_path in val_imgs:
        case_id = get_case_id(img_path)
        if case_id in lab_map:
            # Apply lesion size filter if enabled
            if valid_cases is not None and case_id not in valid_cases:
                continue
            val_labels.append(lab_map[case_id])
            val_valid.append(img_path)
    
    print(f"[INFO] Train samples: {len(train_valid)}, Val samples: {len(val_valid)}")
    
    # Debug info if no samples found
    if len(train_valid) == 0:
        print(f"[WARNING] No training samples found!")
        print(f"  Images in {paths.images_tr}: {len(train_imgs)}")
        if train_imgs:
            sample_img = train_imgs[0]
            sample_case_id = get_case_id(sample_img)
            print(f"  Sample image: {sample_img.name}")
            print(f"  Extracted case ID: '{sample_case_id}'")
            print(f"  Available label IDs (first 5): {list(lab_map.keys())[:5]}")
            print(f"  Case ID in lab_map: {sample_case_id in lab_map}")
    
    if len(val_valid) == 0:
        print(f"[WARNING] No validation samples found!")
        print(f"  Images in {paths.images_val}: {len(val_imgs)}")
    
    train_ds = TumorDataset(train_valid, train_labels, pre_transform=pre_transform)
    val_ds = TumorDataset(val_valid, val_labels, pre_transform=pre_transform)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    auc: float
    confusion_matrix: np.ndarray
    classification_report: str
    predictions: np.ndarray
    targets: np.ndarray
    probabilities: np.ndarray


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> ClassificationMetrics:
    """Evaluate model on a dataset."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])  # Prob of class 1 (malignant)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=["Benign", "Malignant"])
    
    return ClassificationMetrics(
        accuracy=accuracy,
        auc=auc,
        confusion_matrix=cm,
        classification_report=report,
        predictions=all_preds,
        targets=all_targets,
        probabilities=all_probs,
    )


def train_classification_head(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train the classification head (or full model if encoder unfrozen).
    
    Returns:
        Dict with training history and final metrics
    """
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=learning_rate * 0.01,
    )
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
    }
    
    best_auc = 0.0
    best_epoch = 0
    
    print(f"\n{'='*60}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            if (batch_idx + 1) % max(1, len(train_loader) // 3) == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = evaluate_model(model, val_loader, device)
        
        # Compute val loss
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update history
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_metrics.accuracy)
        history["val_auc"].append(val_metrics.auc)
        
        # Learning rate step
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
        print(f"{'='*60}")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_metrics.accuracy:.4f}")
        print(f"Val AUC:    {val_metrics.auc:.4f}")
        print(f"LR:         {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_metrics.auc > best_auc:
            best_auc = val_metrics.auc
            best_epoch = epoch + 1
            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = output_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_metrics.auc,
                    "val_acc": val_metrics.accuracy,
                }, checkpoint_path)
                print(f"✓ Best model saved (AUC: {best_auc:.4f})")
        
        print()
    
    # Final evaluation
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best Epoch: {best_epoch} | Best Val AUC: {best_auc:.4f}")
    print(f"Using final epoch model for evaluation")
    
    # Evaluate using final epoch model (not best checkpoint)
    final_metrics = evaluate_model(model, val_loader, device)
    
    print(f"\n{'='*60}")
    print("Final Validation Metrics")
    print(f"{'='*60}")
    print(f"Accuracy: {final_metrics.accuracy:.4f}")
    print(f"AUC:      {final_metrics.auc:.4f}")
    print(f"\nConfusion Matrix:\n{final_metrics.confusion_matrix}")
    print(f"\nClassification Report:\n{final_metrics.classification_report}")
    
    return {
        "history": history,
        "final_metrics": final_metrics,
        "best_epoch": best_epoch,
        "best_auc": best_auc,
    }


def run_classification_head_experiment(
    paths: Sam3DPaths,
    lab_map: dict,
    *,
    sam3d_root: Optional[Path] = None,
    model_type: str = "vit_b_ori",
    checkpoint: Optional[Path] = None,
    img_size: int = 128,
    device: Optional[str] = None,
    freeze_encoder: bool = True,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    num_workers: int = 2,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run complete classification head experiment on prepared SAM-Med3D data.
    
    This function:
    1. Builds SAM-Med3D model
    2. Adds classification head
    3. Prepares dataloaders
    4. Trains the model
    5. Evaluates and saves results
    
    Args:
        paths: Prepared SAM-Med3D paths
        lab_map: Dictionary mapping case_id to label {0: benign, 1: malignant}
        sam3d_root: Path to SAM-Med3D repo (auto-detected if None)
        model_type: SAM-Med3D model architecture
        checkpoint: Path to pre-trained SAM-Med3D checkpoint
        img_size: Input volume size (default 128)
        device: 'cuda' or 'cpu' (auto-detected if None)
        freeze_encoder: If True, only train classification head; if False, fine-tune encoder
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
        dropout: Dropout rate in classification head
        num_workers: Number of dataloader workers
        output_dir: Directory to save outputs
    
    Returns:
        Dictionary with training history and metrics
    """
    # Setup
    sam3d_root = sam3d_root or find_default_sam3d_root()
    
    if device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    
    print(f"\n{'='*60}")
    print("SAM-Med3D Classification Head Experiment")
    print(f"{'='*60}")
    print(f"Device: {torch_device}")
    print(f"Model: {model_type}")
    print(f"Encoder frozen: {freeze_encoder}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    # Build SAM-Med3D model
    print("[1/4] Building SAM-Med3D model...")
    sam_model = build_sam3d_model(
        sam3d_root=sam3d_root,
        model_type=model_type,
        checkpoint=checkpoint,
        device=torch_device,
        eval_mode=False,  # Training mode
    )
    
    # Add classification head
    print("[2/4] Adding classification head...")
    model = SAMWithClassificationHead(
        sam_model=sam_model,
        num_classes=2,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
    )
    
    # Prepare dataloaders
    print("[3/4] Preparing dataloaders...")
    train_loader, val_loader = prepare_dataloaders(
        paths=paths,
        lab_map=lab_map,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
    )
    
    # Train
    print("[4/4] Starting training...")
    results = train_classification_head(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch_device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        output_dir=output_dir,
    )
    
    # Save results
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save history
        np.save(output_dir / "history.npy", results["history"])
        
        # Save final metrics
        metrics = results["final_metrics"]
        np.save(output_dir / "predictions.npy", metrics.predictions)
        np.save(output_dir / "targets.npy", metrics.targets)
        np.save(output_dir / "probabilities.npy", metrics.probabilities)
        
        # Save summary
        with open(output_dir / "summary.txt", "w") as f:
            f.write(f"SAM-Med3D Classification Head Experiment\n")
            f.write(f"{'='*60}\n")
            f.write(f"Model: {model_type}\n")
            f.write(f"Encoder frozen: {freeze_encoder}\n")
            f.write(f"Best epoch: {results['best_epoch']}\n")
            f.write(f"Best AUC: {results['best_auc']:.4f}\n")
            f.write(f"Final Accuracy: {metrics.accuracy:.4f}\n")
            f.write(f"Final AUC: {metrics.auc:.4f}\n")
            f.write(f"\nConfusion Matrix:\n{metrics.confusion_matrix}\n")
            f.write(f"\nClassification Report:\n{metrics.classification_report}\n")
        
        print(f"\n✓ Results saved to {output_dir}")
    
    return results
