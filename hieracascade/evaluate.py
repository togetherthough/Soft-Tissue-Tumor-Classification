"""Evaluation script for trained models"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
from typing import Dict, List

from hieracascade.models import build_stage1_model, build_stage2_model
from hieracascade.dataio import (
    DatasetStage1,
    DatasetStage2,
    load_labels_csv,
    create_site_held_out_splits,
    FINE_TO_IDX,
    COARSE_TO_IDX
)
from hieracascade.metrics import (
    compute_metrics,
    compute_per_site_metrics,
    print_classification_report,
    compute_confusion_matrix
)
from hieracascade.visualization import (
    plot_confusion_matrix,
    save_saliency_overlay,
    visualize_crop_locations,
    create_prediction_summary
)


@torch.no_grad()
def evaluate_stage1(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    output_dir: Path,
    save_visualizations: bool = True
) -> Dict:
    """Evaluate Stage-1 model"""
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    all_sites = []
    all_study_ids = []
    
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    for batch_idx, (volume, y_fine, y_coarse, modality_id, site, study_id) in enumerate(tqdm(dataloader, desc="Evaluating")):
        volume = volume.to(device)
        modality_id = modality_id.to(device)
        
        # Forward
        logits, saliency = model(volume, modality_id)
        
        # Predictions
        y_pred = torch.argmax(logits, dim=1)
        y_prob = torch.softmax(logits, dim=1)
        
        # Collect
        all_y_true.extend(y_fine.cpu().numpy())
        all_y_pred.extend(y_pred.cpu().numpy())
        all_y_prob.append(y_prob.cpu().numpy())
        all_sites.extend(site)
        all_study_ids.extend(study_id)
        
        # Save visualizations
        if save_visualizations and batch_idx < 10:
            for i in range(volume.shape[0]):
                vol = volume[i, 0].cpu().numpy()
                sal = saliency[i, 0].cpu().numpy()
                
                save_path = vis_dir / f'saliency_{study_id[i]}.png'
                save_saliency_overlay(vol, sal, str(save_path))
    
    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.concatenate(all_y_prob, axis=0)
    all_sites = np.array(all_sites)
    
    # Compute metrics
    class_names = list(FINE_TO_IDX.keys())
    
    print("\n=== Overall Metrics ===")
    metrics = compute_metrics(all_y_true, all_y_pred, all_y_prob, class_names=class_names)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n=== Classification Report ===")
    print_classification_report(all_y_true, all_y_pred, class_names)
    
    print("\n=== Per-Site Metrics ===")
    site_metrics = compute_per_site_metrics(all_y_true, all_y_pred, all_sites, all_y_prob)
    for site, site_m in site_metrics.items():
        print(f"\n{site}:")
        for k, v in site_m.items():
            if 'f1' in k or 'accuracy' in k:
                print(f"  {k}: {v:.4f}")
    
    # Confusion matrix
    cm = compute_confusion_matrix(all_y_true, all_y_pred, normalize='true')
    plot_confusion_matrix(
        cm, class_names,
        output_path=str(output_dir / 'confusion_matrix.png'),
        title='Stage-1 Confusion Matrix (Normalized)'
    )
    
    # Save predictions
    create_prediction_summary(
        all_study_ids, all_y_true, all_y_pred, all_y_prob,
        class_names, str(output_dir / 'predictions.csv')
    )
    
    return metrics


@torch.no_grad()
def evaluate_stage2(
    model: nn.Module,
    stage1_model: nn.Module,
    dataloader: DataLoader,
    device: str,
    output_dir: Path
) -> Dict:
    """Evaluate Stage-2 model"""
    model.eval()
    stage1_model.eval()
    
    all_y_true_fine = []
    all_y_pred_fine = []
    all_y_prob_fine = []
    all_y_true_coarse = []
    all_y_pred_coarse = []
    all_y_prob_coarse = []
    all_sites = []
    all_study_ids = []
    
    for crops, y_fine, y_coarse, modality_id, site, study_id in tqdm(dataloader, desc="Evaluating"):
        crops = crops.to(device)
        modality_id = modality_id.to(device)
        
        # Forward
        logits_fine, logits_coarse, _ = model(crops, modality_id)
        
        # Predictions
        y_pred_fine = torch.argmax(logits_fine, dim=1)
        y_pred_coarse = torch.argmax(logits_coarse, dim=1)
        y_prob_fine = torch.softmax(logits_fine, dim=1)
        y_prob_coarse = torch.softmax(logits_coarse, dim=1)
        
        # Collect
        all_y_true_fine.extend(y_fine.cpu().numpy())
        all_y_pred_fine.extend(y_pred_fine.cpu().numpy())
        all_y_prob_fine.append(y_prob_fine.cpu().numpy())
        all_y_true_coarse.extend(y_coarse.cpu().numpy())
        all_y_pred_coarse.extend(y_pred_coarse.cpu().numpy())
        all_y_prob_coarse.append(y_prob_coarse.cpu().numpy())
        all_sites.extend(site)
        all_study_ids.extend(study_id)
    
    # Convert to arrays
    all_y_true_fine = np.array(all_y_true_fine)
    all_y_pred_fine = np.array(all_y_pred_fine)
    all_y_prob_fine = np.concatenate(all_y_prob_fine, axis=0)
    all_y_true_coarse = np.array(all_y_true_coarse)
    all_y_pred_coarse = np.array(all_y_pred_coarse)
    all_y_prob_coarse = np.concatenate(all_y_prob_coarse, axis=0)
    all_sites = np.array(all_sites)
    
    # Fine-grained metrics
    fine_class_names = list(FINE_TO_IDX.keys())
    coarse_class_names = list(COARSE_TO_IDX.keys())
    
    print("\n=== Fine-Grained Metrics ===")
    metrics_fine = compute_metrics(all_y_true_fine, all_y_pred_fine, all_y_prob_fine, class_names=fine_class_names)
    for k, v in metrics_fine.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n=== Fine-Grained Classification Report ===")
    print_classification_report(all_y_true_fine, all_y_pred_fine, fine_class_names)
    
    # Coarse metrics
    print("\n=== Coarse (Family) Metrics ===")
    metrics_coarse = compute_metrics(all_y_true_coarse, all_y_pred_coarse, all_y_prob_coarse, class_names=coarse_class_names)
    for k, v in metrics_coarse.items():
        print(f"  {k}: {v:.4f}")
    
    # Per-site metrics (fine)
    print("\n=== Per-Site Fine-Grained Metrics ===")
    site_metrics = compute_per_site_metrics(all_y_true_fine, all_y_pred_fine, all_sites, all_y_prob_fine)
    for site, site_m in site_metrics.items():
        print(f"\n{site}:")
        for k, v in site_m.items():
            if 'f1' in k or 'accuracy' in k:
                print(f"  {k}: {v:.4f}")
    
    # Confusion matrices
    cm_fine = compute_confusion_matrix(all_y_true_fine, all_y_pred_fine, normalize='true')
    plot_confusion_matrix(
        cm_fine, fine_class_names,
        output_path=str(output_dir / 'confusion_matrix_fine.png'),
        title='Fine-Grained Confusion Matrix (Normalized)'
    )
    
    cm_coarse = compute_confusion_matrix(all_y_true_coarse, all_y_pred_coarse, normalize='true')
    plot_confusion_matrix(
        cm_coarse, coarse_class_names,
        output_path=str(output_dir / 'confusion_matrix_coarse.png'),
        title='Coarse (Family) Confusion Matrix (Normalized)'
    )
    
    # Save predictions
    create_prediction_summary(
        all_study_ids, all_y_true_fine, all_y_pred_fine, all_y_prob_fine,
        fine_class_names, str(output_dir / 'predictions_fine.csv')
    )
    
    create_prediction_summary(
        all_study_ids, all_y_true_coarse, all_y_pred_coarse, all_y_prob_coarse,
        coarse_class_names, str(output_dir / 'predictions_coarse.csv')
    )
    
    return metrics_fine


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--stage', type=str, required=True, choices=['stage1', 'stage2'], help='Which stage')
    parser.add_argument('--data_root', type=str, required=True, help='Root data directory')
    parser.add_argument('--labels_csv', type=str, required=True, help='Labels CSV')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--stage1_ckpt', type=str, help='Stage-1 checkpoint (for Stage-2 evaluation)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint['config']
    
    # Load data
    index = load_labels_csv(args.labels_csv)
    splits = create_site_held_out_splits(index)
    _, val_index = splits[args.fold]
    
    if args.stage == 'stage1':
        # Build model
        model = build_stage1_model(
            n_classes=len(FINE_TO_IDX),
            backbone=config['model']['name'].replace('stage1_', ''),
            embed_dim=config['model']['embed_dim']
        ).to(args.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create dataset
        dataset = DatasetStage1(
            val_index,
            data_root=args.data_root,
            target_spacing=config['geom']['spacing'],
            target_size=tuple(config['geom']['size']),
            augment=False
        )
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)
        
        # Evaluate
        evaluate_stage1(model, dataloader, args.device, output_dir)
    
    else:  # stage2
        if not args.stage1_ckpt:
            raise ValueError("--stage1_ckpt required for Stage-2 evaluation")
        
        # Load Stage-1
        stage1_ckpt = torch.load(args.stage1_ckpt, map_location=args.device)
        stage1_model = build_stage1_model(
            n_classes=len(FINE_TO_IDX),
            backbone=stage1_ckpt['config']['model']['name'].replace('stage1_', ''),
            embed_dim=stage1_ckpt['config']['model']['embed_dim']
        ).to(args.device)
        stage1_model.load_state_dict(stage1_ckpt['model_state_dict'])
        stage1_model.eval()
        
        # Build Stage-2 model
        model = build_stage2_model(
            n_fine=len(FINE_TO_IDX),
            n_coarse=len(COARSE_TO_IDX),
            backbone=config['model']['name'].replace('stage2_', ''),
            embed_dim=config['model']['emb_dim'],
            pooling=config['model']['pool']
        ).to(args.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create datasets
        stage1_dataset = DatasetStage1(
            val_index,
            data_root=args.data_root,
            target_spacing=config['geom']['spacing'],
            target_size=tuple(config['geom']['size_stage1']),
            augment=False
        )
        
        dataset = DatasetStage2(
            stage1_dataset,
            stage1_model,
            K=config['proposals']['K'],
            crop_size=config['proposals']['crop_size'],
            nms_distance=config['proposals']['nms_dist'],
            top_p=config['proposals']['top_p'],
            device=args.device
        )
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        
        # Evaluate
        evaluate_stage2(model, stage1_model, dataloader, args.device, output_dir)
    
    print(f"\n✓ Evaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
