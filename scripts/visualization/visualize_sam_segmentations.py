"""
Visualize SAM-Med3D segmentation results to understand low Dice scores.
Shows CT slice, ground truth, SAM prediction, and overlay comparison.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from med3pipe.sam.core import build_sam3d_model, find_default_sam3d_root
from importlib.machinery import SourceFileLoader
loader = SourceFileLoader('sam_dice_script', str(project_root / 'scripts' / 'compute_sam_dice_scores.py'))
mod = loader.load_module()


def visualize_case(
    case_info: Dict,
    sam_predictor,
    save_dir: Path,
    num_slices: int = 5
):
    """Visualize SAM segmentation for a single case."""
    
    case_id = case_info['case_id']
    print(f"\nProcessing: {case_id}")
    
    # Load and preprocess
    image_sitk = mod.merge_images(case_info['image_files'])
    mask_sitk = mod.merge_segmentations(case_info['segmentation_files'])
    
    image_tensor = mod.load_volume_for_sam(image_sitk, img_size=128, modality='CT')
    gt_mask = mod.load_mask_for_sam(mask_sitk, img_size=128)
    
    print(f"  Image shape: {tuple(image_tensor.shape)}")
    print(f"  GT voxels: {gt_mask.sum()}")
    
    # Get click point (center of GT mask)
    coords = np.argwhere(gt_mask > 0)
    if len(coords) == 0:
        print(f"  [WARNING] Empty GT mask, skipping")
        return None
    
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    center_z = (z_min + z_max) // 2
    
    # Run SAM inference
    device = next(sam_predictor.parameters()).device
    sam_predictor.eval()
    
    with torch.no_grad():
        # Get image embeddings
        input_tensor = image_tensor.to(device)
        image_embeddings = sam_predictor.image_encoder(input_tensor)
        
        D, H, W = input_tensor.shape[2:]
        
        # Prepare point prompt
        point = torch.tensor([[[center_x, center_y, center_z]]], device=device, dtype=torch.float)
        label = torch.tensor([[1]], device=device, dtype=torch.int64)
        
        # Encode prompts
        sparse_embeddings, dense_embeddings = sam_predictor.prompt_encoder(
            points=[point, label],
            boxes=None,
            masks=None
        )
        
        # Decode mask
        low_res_masks, _ = sam_predictor.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam_predictor.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        
        # Upsample to original resolution
        final_masks = torch.nn.functional.interpolate(
            low_res_masks, 
            size=(D, H, W), 
            mode='trilinear', 
            align_corners=False
        )
        sam_mask = (torch.sigmoid(final_masks) > 0.5).cpu().squeeze().numpy()
    
    # Compute metrics
    intersection = np.logical_and(gt_mask, sam_mask).sum()
    union = np.logical_or(gt_mask, sam_mask).sum()
    dice = (2 * intersection) / (gt_mask.sum() + sam_mask.sum()) if (gt_mask.sum() + sam_mask.sum()) > 0 else 0
    iou = intersection / union if union > 0 else 0
    
    print(f"  SAM voxels: {sam_mask.sum()}")
    print(f"  Dice: {dice:.4f}")
    print(f"  IoU: {iou:.4f}")
    
    # Select slices to visualize (around the lesion)
    z_coords = coords[:, 0]
    z_min, z_max = z_coords.min(), z_coords.max()
    z_center = (z_min + z_max) // 2
    
    # Distribute slices evenly across the lesion
    z_indices = np.linspace(z_min, z_max, num_slices, dtype=int)
    
    # Create visualization
    fig, axes = plt.subplots(num_slices, 4, figsize=(16, num_slices * 3))
    if num_slices == 1:
        axes = axes.reshape(1, -1)
    
    for idx, z in enumerate(z_indices):
        # Get slices
        ct_slice = image_tensor[0, 0, z].cpu().numpy()
        gt_slice = gt_mask[z]
        sam_slice = sam_mask[z]
        
        # CT slice
        ax = axes[idx, 0]
        vmin, vmax = np.percentile(ct_slice[ct_slice > -900], [1, 99])
        ax.imshow(ct_slice, cmap='gray', vmin=vmin, vmax=vmax)
        if z == center_z:
            ax.plot(center_x, center_y, 'r*', markersize=15, label='Click')
            ax.legend(loc='upper right')
        ax.set_title(f'CT Slice Z={z}')
        ax.axis('off')
        
        # Ground truth
        ax = axes[idx, 1]
        ax.imshow(ct_slice, cmap='gray', vmin=vmin, vmax=vmax, alpha=0.5)
        ax.imshow(gt_slice, cmap='Greens', alpha=0.5, vmin=0, vmax=1)
        ax.set_title('Ground Truth')
        ax.axis('off')
        
        # SAM prediction
        ax = axes[idx, 2]
        ax.imshow(ct_slice, cmap='gray', vmin=vmin, vmax=vmax, alpha=0.5)
        ax.imshow(sam_slice, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
        ax.set_title('SAM Prediction')
        ax.axis('off')
        
        # Overlay comparison
        ax = axes[idx, 3]
        ax.imshow(ct_slice, cmap='gray', vmin=vmin, vmax=vmax, alpha=0.3)
        
        # Show TP (white), FP (red), FN (blue)
        gt_bool = gt_slice.astype(bool)
        sam_bool = sam_slice.astype(bool)
        tp = np.logical_and(gt_bool, sam_bool)
        fp = np.logical_and(~gt_bool, sam_bool)
        fn = np.logical_and(gt_bool, ~sam_bool)
        
        overlay = np.zeros((*ct_slice.shape, 3))
        overlay[tp] = [1, 1, 1]  # White - correct
        overlay[fp] = [1, 0, 0]  # Red - false positive
        overlay[fn] = [0, 0, 1]  # Blue - false negative
        
        ax.imshow(overlay, alpha=0.6)
        ax.set_title('Overlap (W=TP, R=FP, B=FN)')
        ax.axis('off')
    
    # Add overall title
    fig.suptitle(
        f'{case_id} | Dice: {dice:.4f} | IoU: {iou:.4f} | GT: {gt_mask.sum()}, SAM: {sam_mask.sum()}',
        fontsize=14, fontweight='bold'
    )
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='white', label='True Positive'),
        mpatches.Patch(color='red', label='False Positive'),
        mpatches.Patch(color='blue', label='False Negative'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save
    output_file = save_dir / f'{case_id}_sam_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved: {output_file.name}")
    
    return {
        'case_id': case_id,
        'dice': dice,
        'iou': iou,
        'gt_voxels': int(gt_mask.sum()),
        'sam_voxels': int(sam_mask.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize SAM segmentation results')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to visualize (e.g., gist, lipo, crlm)')
    parser.add_argument('--num_cases', type=int, default=5,
                        help='Number of cases to visualize')
    parser.add_argument('--num_slices', type=int, default=5,
                        help='Number of slices per case')
    parser.add_argument('--config', type=str, default='configs/datasets_analysis.yaml',
                        help='Config file path')
    parser.add_argument('--output_dir', type=str, default='results/sam_visualization',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Setup paths
    config_path = project_root / args.config
    output_dir = project_root / args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"SAM-Med3D Segmentation Visualization")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load SAM model
    print("Loading SAM-Med3D model...")
    sam3d_root = find_default_sam3d_root()
    checkpoint_path = sam3d_root / 'ckpt' / 'sam_med3d_turbo.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sam_predictor = build_sam3d_model(
        sam3d_root=sam3d_root,
        model_type="vit_b_ori",
        checkpoint=checkpoint_path,
        device=device,
        eval_mode=True
    )
    print(f"[OK] Model loaded on {device}")
    
    # Discover cases
    print(f"\nDiscovering cases from config...")
    datasets = mod.discover_cases_from_config(config_path, project_root)
    
    if args.dataset not in datasets:
        print(f"\n[ERROR] Dataset '{args.dataset}' not found in config")
        print(f"Available datasets: {list(datasets.keys())}")
        return
    
    cases = datasets[args.dataset]['cases']
    print(f"Found {len(cases)} cases")
    
    # Select cases to visualize
    num_to_viz = min(args.num_cases, len(cases))
    selected_cases = cases[:num_to_viz]
    
    print(f"Visualizing {num_to_viz} cases...")
    
    # Process each case
    results = []
    for case_info in selected_cases:
        try:
            result = visualize_case(
                case_info,
                sam_predictor,
                output_dir,
                num_slices=args.num_slices
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"  [ERROR] Error: {e}")
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Successfully visualized: {len(results)}/{num_to_viz} cases")
    
    if results:
        avg_dice = np.mean([r['dice'] for r in results])
        avg_iou = np.mean([r['iou'] for r in results])
        print(f"\nAverage Dice: {avg_dice:.4f}")
        print(f"Average IoU:  {avg_iou:.4f}")
        
        print(f"\nPer-case results:")
        print(f"{'Case':<30} {'Dice':>8} {'IoU':>8} {'GT Voxels':>12} {'SAM Voxels':>12}")
        print("-" * 80)
        for r in results:
            print(f"{r['case_id']:<30} {r['dice']:>8.4f} {r['iou']:>8.4f} {r['gt_voxels']:>12} {r['sam_voxels']:>12}")
    
    print(f"\n[OK] All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
