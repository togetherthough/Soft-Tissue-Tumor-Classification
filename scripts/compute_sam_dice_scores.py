"""
Compute Dice scores for all images using SAM-Med3D with 11-point iterative refinement.

This script:
- Finds all available datasets in SAM-Med3D-main/data/
- Uses proper ResizeLargestTo + ZNormalization preprocessing (no CropOrPad for actual resizing)
- Applies 11-click iterative refinement matching SAM-Med3D training
- Computes Dice scores for each image
- Saves results to CSV
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import SimpleITK as sitk
import torchio as tio
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from med3pipe.sam.core import build_sam3d_model, find_default_sam3d_root
from med3pipe.sam.transforms import ResizeLargestTo
from med3pipe.data.prepare import (
    find_case_dirs,
    find_image_files,
    find_segmentation_files,
    merge_images,
    merge_segmentations,
)


def _znorm_masking_method(x):
    """Masking method for ZNormalization."""
    return x > 0


def load_volume_for_sam(img_source, img_size=128, modality: str | None = None):
    """
    Load volume with proper preprocessing for SAM.
    
    Uses ResizeLargestTo (preserves aspect ratio) + minimal CropOrPad + Z-Normalization.
    This is the recommended approach per PREPROCESSING_UPGRADE.md.
    """
    if isinstance(img_source, sitk.Image):
        sitk_img = img_source
    else:
        sitk_img = sitk.ReadImage(str(img_source))
    sitk_arr, _ = tio.data.io.sitk_to_nib(sitk_img)
    subject = tio.Subject(image=tio.ScalarImage(tensor=sitk_arr))
    
    # Apply CT clamping if modality is CT or path indicates CT
    modality_str = (modality or "").upper()
    if modality_str == "CT":
        subject = tio.Clamp(-1000, 1000)(subject)
    elif not modality and isinstance(img_source, (str, Path)):
        src_str = str(img_source).lower()
        if "/ct_" in src_str or "_ct" in src_str:
            subject = tio.Clamp(-1000, 1000)(subject)
    elif isinstance(img_source, Path):
        src_str = str(img_source).lower()
        if "/ct_" in src_str or "_ct" in src_str:
            subject = tio.Clamp(-1000, 1000)(subject)
    
    # Resize-then-pad + NORMALIZATION (critical for SAM!)
    transform = tio.Compose([
        tio.ToCanonical(),
        ResizeLargestTo(target_size=img_size),
        tio.CropOrPad(target_shape=(img_size, img_size, img_size)),  # Minimal padding after resize
        tio.ZNormalization(masking_method=_znorm_masking_method),  # CRITICAL for SAM!
    ])
    subject = transform(subject)
    
    image = subject.image.data.clone().detach()
    image = image.unsqueeze(0)  # (1, 1, D, H, W)
    image = image.float()
    return image


def load_mask_for_sam(mask_source, img_size=128):
    """Load mask with resize-then-pad preprocessing."""
    if isinstance(mask_source, sitk.Image):
        sitk_mask = mask_source
    else:
        sitk_mask = sitk.ReadImage(str(mask_source))
    mask_arr, _ = tio.data.io.sitk_to_nib(sitk_mask)
    subject = tio.Subject(label=tio.LabelMap(tensor=mask_arr))
    
    transform = tio.Compose([
        tio.ToCanonical(),
        ResizeLargestTo(target_size=img_size),
        tio.CropOrPad(target_shape=(img_size, img_size, img_size)),  # Minimal padding after resize
    ])
    subject = transform(subject)
    
    mask = subject.label.data.squeeze().numpy()
    mask = (mask > 0).astype(float)
    return mask


def infer_case_id(case_dir: Path) -> str:
    """Infer case identifier from a NIFTI directory path."""
    case_dir = case_dir.resolve()
    if case_dir.name.lower() == "nifti":
        parent = case_dir.parent
        grandparent = parent.parent if parent else None
        if grandparent and grandparent.name:
            return grandparent.name
        if parent.name:
            return parent.name
    return case_dir.name


def discover_cases_from_config(config_path: Path, project_root: Path) -> dict:
    """
    Discover all dataset cases directly from the raw dataset roots defined in the config.
    
    Returns:
        dict mapping dataset name -> dataset info with case metadata.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    datasets_cfg = config.get("datasets", {})
    discovered = {}
    for dataset_name, cfg in datasets_cfg.items():
        dataset_root = Path(cfg.get("dataset_root", "")).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = (project_root / dataset_root).resolve()
        else:
            dataset_root = dataset_root.resolve()
        if not dataset_root.exists():
            print(f"⚠️  Dataset root missing for {dataset_name}: {dataset_root}")
            continue
        case_glob = cfg.get("prepare", {}).get("case_glob")
        case_dirs = find_case_dirs(dataset_root, case_glob=case_glob)
        cases = []
        for case_dir in case_dirs:
            image_files = find_image_files(case_dir)
            seg_files = find_segmentation_files(case_dir)
            if not image_files:
                print(f"   ⚠️  Skipping {case_dir} (no image files)")
                continue
            if not seg_files:
                print(f"   ⚠️  Skipping {case_dir} (no segmentation files)")
                continue
            case_id = infer_case_id(case_dir)
            cases.append({
                "case_id": case_id,
                "image_files": image_files,
                "segmentation_files": seg_files,
            })
        if not cases:
            print(f"⚠️  No valid cases discovered for {dataset_name} in {dataset_root}")
            continue
        discovered[dataset_name] = {
            "dataset": dataset_name,
            "category": cfg.get("category", dataset_name),
            "ct_name": cfg.get("ct_name", dataset_name),
            "modality": cfg.get("modality", "CT"),
            "cases": cases,
        }
    return discovered


def compute_dice_score(pred_mask, gt_mask):
    """Compute Dice coefficient between prediction and ground truth."""
    pred_mask = (pred_mask > 0).astype(np.float32)
    gt_mask = (gt_mask > 0).astype(np.float32)
    
    intersection = np.sum(pred_mask * gt_mask)
    pred_sum = np.sum(pred_mask)
    gt_sum = np.sum(gt_mask)
    
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    
    dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)
    return dice


def sample_next_click(prev_mask, gt_mask):
    """
    Sample next click point based on prediction errors.
    
    Key improvement: Bias toward false negatives (FN) when there's class imbalance.
    When prediction is spatially misaligned, FP region can be 10-100× larger than FN.
    Sampling FN helps the model discover the correct target location.
    """
    if isinstance(prev_mask, np.ndarray):
        prev_mask = torch.from_numpy(prev_mask)
    if isinstance(gt_mask, np.ndarray):
        gt_mask = torch.from_numpy(gt_mask)
    
    if prev_mask.ndim != 3:
        prev_mask = prev_mask.squeeze()
    if gt_mask.ndim != 3:
        gt_mask = gt_mask.squeeze()
    
    prev_mask = prev_mask > 0.5
    gt_mask = gt_mask > 0
    
    fn_mask = torch.logical_and(gt_mask, torch.logical_not(prev_mask))
    fp_mask = torch.logical_and(torch.logical_not(gt_mask), prev_mask)
    
    fn_points = torch.argwhere(fn_mask)
    fp_points = torch.argwhere(fp_mask)
    
    fn_count = len(fn_points)
    fp_count = len(fp_points)
    
    point = None
    is_positive = False
    
    if fn_count > 0 and fp_count > 0:
        # Bias toward false negatives when there's severe class imbalance
        # This helps the model find the correct target location
        if fp_count > fn_count * 5:
            # Severe imbalance: heavily favor FN sampling (90%)
            prob_fn = 0.9
        elif fp_count > fn_count * 2:
            # Moderate imbalance: favor FN sampling (80%)
            prob_fn = 0.8
        else:
            # Mild imbalance: still favor FN to help localization (70%)
            prob_fn = 0.7
        
        if np.random.random() < prob_fn:
            point = fn_points[np.random.randint(fn_count)]
            is_positive = True
        else:
            point = fp_points[np.random.randint(fp_count)]
            is_positive = False
    elif fn_count > 0:
        point = fn_points[np.random.randint(fn_count)]
        is_positive = True
    elif fp_count > 0:
        point = fp_points[np.random.randint(fp_count)]
        is_positive = False
    else:
        gt_points = torch.argwhere(gt_mask)
        if len(gt_points) > 0:
            point = gt_points[np.random.randint(len(gt_points))]
            is_positive = True
        else:
            point = torch.tensor([gt_mask.shape[0] // 2, gt_mask.shape[1] // 2, gt_mask.shape[2] // 2])
            is_positive = False
    
    point_xyz = torch.tensor([point[2], point[1], point[0]], dtype=torch.float32)
    point_tensor = point_xyz.reshape(1, 1, 3)
    label_tensor = torch.tensor([[int(is_positive)]], dtype=torch.int64)
    
    return point_tensor, label_tensor


def generate_sam_with_iterative_refinement(
    model,
    image_tensor,
    gt_mask_tensor,
    device,
    num_clicks=11,
    threshold=0.5,
    verbose=False,
    return_debug=False,
):
    """
    Generate segmentation using iterative refinement with multiple clicks.
    Matches the SAM-Med3D training procedure.
    """
    model.eval()
    
    debug_info = {
        'per_click_dice': [],
        'per_click_points': [],
        'per_click_labels': [],
        'per_click_voxels': [],
        'final_probs': {},
    }

    with torch.no_grad():
        input_tensor = image_tensor.to(device)
        
        # Get image embeddings (computed once)
        image_embeddings = model.image_encoder(input_tensor)
        
        D, H, W = input_tensor.shape[2:]
        
        # Initialize
        prev_mask = torch.zeros((D, H, W), device=device)
        low_res_mask = torch.zeros((1, 1, D//4, H//4, W//4), device=device, dtype=torch.float)
        
        # Store all points for multi-click prompting
        all_points = []
        all_labels = []
        
        # Iterative refinement loop
        for click_idx in range(num_clicks):
            # Generate next click based on current prediction vs ground truth
            if click_idx == 0:
                # First click: use center of ground truth
                coords = np.argwhere(gt_mask_tensor > 0)
                if len(coords) > 0:
                    z_min, y_min, x_min = coords.min(axis=0)
                    z_max, y_max, x_max = coords.max(axis=0)
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2
                    center_z = (z_min + z_max) // 2
                    point = torch.tensor([[[center_x, center_y, center_z]]], device=device, dtype=torch.float)
                    label = torch.tensor([[1]], device=device, dtype=torch.int64)
                else:
                    # Fallback to image center
                    point = torch.tensor([[[W//2, H//2, D//2]]], device=device, dtype=torch.float)
                    label = torch.tensor([[1]], device=device, dtype=torch.int64)
            else:
                # Subsequent clicks: sample from error regions
                point, label = sample_next_click(prev_mask.cpu(), gt_mask_tensor)
                point = point.to(device)
                label = label.to(device)
            
            all_points.append(point)
            all_labels.append(label)
            debug_info['per_click_points'].append(point.detach().cpu().numpy())
            debug_info['per_click_labels'].append(int(label.item()))
            
            # Concatenate all points so far
            points_co = torch.cat(all_points, dim=1)  # [1, num_points, 3]
            points_la = torch.cat(all_labels, dim=1)  # [1, num_points]
            
            # Forward pass with current points
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=[points_co, points_la],
                boxes=None,
                masks=low_res_mask
            )
            
            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            
            # Update for next iteration
            low_res_mask = low_res_masks
            
            # Get high-res prediction for next click sampling
            prev_masks_hr = F.interpolate(low_res_masks, size=(D, H, W), mode='trilinear', align_corners=False)
            prev_mask = torch.sigmoid(prev_masks_hr).squeeze()
            dice_val = compute_dice_score(prev_mask.detach().cpu().numpy(), gt_mask_tensor)
            debug_info['per_click_dice'].append(float(dice_val))
            debug_info['per_click_voxels'].append(int((prev_mask > threshold).sum().item()))
            if verbose:
                print(
                    f"      Click {click_idx + 1:02d}/{num_clicks}: Dice={dice_val:.4f} | "
                    f"Point={tuple(point.flatten().tolist())} | Label={int(label.item())}"
                )
        
        # Final prediction
        final_masks_hr = F.interpolate(low_res_masks, size=(D, H, W), mode='trilinear', align_corners=False)
        seg_prob = torch.sigmoid(final_masks_hr)
        seg_mask = (seg_prob > threshold).cpu().squeeze().numpy().astype(np.uint8)
        debug_info['final_probs'] = {
            'mean': float(seg_prob.mean().item()),
            'max': float(seg_prob.max().item()),
            'min': float(seg_prob.min().item()),
        }
        if verbose:
            final_dice = compute_dice_score(seg_mask, gt_mask_tensor)
            print(
                f"    ✅ Final Dice: {final_dice:.4f} | "
                f"Mask voxels: {int(seg_mask.sum())} | Prob mean/max: {debug_info['final_probs']['mean']:.4f}/"
                f"{debug_info['final_probs']['max']:.4f}"
            )
    
    if return_debug:
        return seg_mask, debug_info
    return seg_mask


def compute_all_dice_scores(
    model,
    datasets,
    device,
    img_size=128,
    num_clicks=11,
    threshold=0.5,
    output_csv=None,
    verbose=False,
    debug_case=None,
    debug_dir=None,
    min_voxels=None,
    min_dimension=None,
):
    """
    Compute dice scores for all images in all datasets.
    
    Args:
        model: SAM-Med3D model
        datasets: dict from discover_cases_from_config()
        device: torch device
        img_size: image size for preprocessing
        num_clicks: number of iterative refinement clicks
        threshold: segmentation threshold
        output_csv: path to save results CSV (optional)
    
    Returns:
        DataFrame with dice scores
    """
    results = []
    
    print(f"\n{'='*80}")
    print(f"Computing Dice Scores for All Images")
    print(f"{'='*80}")
    print(f"Datasets: {len(datasets)}")
    print(f"Iterative refinement clicks: {num_clicks}")
    print(f"Segmentation threshold: {threshold}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    for dataset_name, dataset_info in datasets.items():
        category = dataset_info['category']
        modality = dataset_info.get('modality', 'CT')
        cases = dataset_info['cases']
        
        print(f"\n{'─'*80}")
        print(f"📊 Dataset: {dataset_name}")
        print(f"   Category: {category} | Modality: {modality}")
        print(f"   Cases: {len(cases)}")
        print(f"{'─'*80}")
        
        if len(cases) == 0:
            print("   ⚠️  No cases found, skipping...")
            continue
        
        for case in tqdm(cases, desc=f"  Processing {dataset_name}", unit="case"):
            case_name = case['case_id']
            is_debug_case = (
                debug_case is not None
                and debug_case[0].lower() == dataset_name.lower()
                and debug_case[1].lower() == case_name.lower()
            )
            
            try:
                image_sitk = merge_images(case['image_files'])
                mask_sitk = merge_segmentations(case['segmentation_files'])
                
                image_tensor = load_volume_for_sam(
                    image_sitk,
                    img_size=img_size,
                    modality=modality,
                )
                gt_mask = load_mask_for_sam(mask_sitk, img_size=img_size)
                
                # Apply lesion size filtering if requested
                if min_voxels is not None or min_dimension is not None:
                    gt_voxels = gt_mask.sum()
                    
                    # Check voxel threshold
                    if min_voxels is not None and gt_voxels < min_voxels:
                        if verbose:
                            print(f"  ⏭️  Skipping {case_name}: {gt_voxels:.0f} voxels < {min_voxels} threshold")
                        continue
                    
                    # Check minimum dimension threshold
                    if min_dimension is not None:
                        coords = np.argwhere(gt_mask > 0)
                        if len(coords) > 0:
                            z_min, y_min, x_min = coords.min(axis=0)
                            z_max, y_max, x_max = coords.max(axis=0)
                            min_dim = min(x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1)
                            if min_dim < min_dimension:
                                if verbose:
                                    print(f"  ⏭️  Skipping {case_name}: min dimension {min_dim} < {min_dimension} threshold")
                                continue
                
                # Generate segmentation with iterative refinement
                if is_debug_case:
                    pred_mask, debug_info = generate_sam_with_iterative_refinement(
                        model,
                        image_tensor,
                        gt_mask,
                        device,
                        num_clicks=num_clicks,
                        threshold=threshold,
                        verbose=True,
                        return_debug=True,
                    )
                else:
                    pred_mask = generate_sam_with_iterative_refinement(
                        model,
                        image_tensor,
                        gt_mask,
                        device,
                        num_clicks=num_clicks,
                        threshold=threshold,
                        verbose=verbose,
                        return_debug=False,
                    )
                    debug_info = None
                
                # Compute dice score
                dice = compute_dice_score(pred_mask, gt_mask)
                
                results.append({
                    'dataset': category,
                    'split': dataset_name,
                    'case': case_name,
                    'dice_score': dice,
                    'num_clicks': num_clicks,
                    'threshold': threshold,
                    'gt_voxels': gt_mask.sum(),
                    'pred_voxels': pred_mask.sum()
                })
                
                if is_debug_case:
                    print(f"\n🔍 Debug info for {dataset_name}/{case_name}:")
                    print("  Per-click Dice:", debug_info['per_click_dice'])
                    print("  Per-click voxels:", debug_info['per_click_voxels'])
                    print("  Final prob stats:", debug_info['final_probs'])
                    if debug_dir:
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        debug_path = debug_dir / f"{dataset_name}_{case_name}_debug.json"
                        # Convert numpy arrays to lists for JSON serialization
                        debug_serializable = {
                            'per_click_dice': [float(x) for x in debug_info['per_click_dice']],
                            'per_click_points': [p.tolist() for p in debug_info['per_click_points']],
                            'per_click_labels': [int(x) for x in debug_info['per_click_labels']],
                            'per_click_voxels': [int(x) for x in debug_info['per_click_voxels']],
                            'final_probs': {k: float(v) for k, v in debug_info['final_probs'].items()},
                        }
                        with open(debug_path, 'w') as f:
                            json.dump({
                                'dataset': dataset_name,
                                'case': case_name,
                                'dice_score': float(dice),
                                'gt_voxels': float(gt_mask.sum()),
                                'pred_voxels': float(pred_mask.sum()),
                                'debug': debug_serializable,
                            }, f, indent=2)
                        print(f"  📝 Saved debug log to {debug_path}")
                
            except Exception as e:
                print(f"\n   ❌ Error processing {case_name}: {e}")
                results.append({
                    'dataset': category,
                    'split': dataset_name,
                    'case': case_name,
                    'dice_score': None,
                    'num_clicks': num_clicks,
                    'threshold': threshold,
                    'gt_voxels': None,
                    'pred_voxels': None,
                    'error': str(e)
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"📈 Summary Statistics")
    print(f"{'='*80}")
    
    if 'dice_score' in df.columns and df['dice_score'].notna().sum() > 0:
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            valid_scores = dataset_df['dice_score'].dropna()
            
            if len(valid_scores) > 0:
                print(f"\n{dataset}:")
                print(f"  Cases processed: {len(valid_scores)}")
                print(f"  Mean Dice: {valid_scores.mean():.4f} ± {valid_scores.std():.4f}")
                print(f"  Median Dice: {valid_scores.median():.4f}")
                print(f"  Min Dice: {valid_scores.min():.4f}")
                print(f"  Max Dice: {valid_scores.max():.4f}")
        
        # Overall statistics
        all_valid = df['dice_score'].dropna()
        print(f"\n{'─'*40}")
        print(f"Overall (all datasets):")
        print(f"  Total cases: {len(all_valid)}")
        print(f"  Mean Dice: {all_valid.mean():.4f} ± {all_valid.std():.4f}")
        print(f"  Median Dice: {all_valid.median():.4f}")
        print(f"{'='*80}\n")
    
    # Save to CSV
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"✅ Results saved to: {output_csv}\n")
    
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SAM-Med3D Dice scores")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names to process (default: all)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional limit on number of cases per dataset",
    )
    parser.add_argument(
        "--debug-case",
        type=str,
        default=None,
        help="Specify dataset:case_id to log per-click debug information",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default="results/debug_logs",
        help="Directory to store debug JSON outputs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for all cases",
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=None,
        help="Skip cases where preprocessed GT has fewer than N voxels",
    )
    parser.add_argument(
        "--min-dimension",
        type=int,
        default=None,
        help="Skip cases where smallest bbox dimension is less than N",
    )
    return parser.parse_args()


def main():
    """Main function."""
    # Configuration
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 128
    num_clicks = 11
    threshold = 0.5
    config_path = project_root / "configs" / "datasets_analysis.yaml"
    
    # SAM-Med3D checkpoint
    checkpoint_path = project_root / "SAM-Med3D-main" / "SAM-Med3D-main" / "ckpt" / "sam_med3d_turbo.pth"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please download sam_med3d_turbo.pth first.")
        return
    
    # Find SAM-Med3D root
    sam3d_root = find_default_sam3d_root()
    
    print(f"\n{'='*80}")
    print(f"SAM-Med3D Dice Score Computation")
    print(f"{'='*80}")
    print(f"Project root: {project_root}")
    print(f"SAM-Med3D root: {sam3d_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Load SAM-Med3D model
    print("Loading SAM-Med3D model...")
    model = build_sam3d_model(
        sam3d_root=sam3d_root,
        model_type="vit_b_ori",
        checkpoint=checkpoint_path,
        device=device,
        eval_mode=True
    )
    print("✅ Model loaded successfully!\n")
    
    # Discover raw dataset cases from config
    if not config_path.exists():
        print(f"❌ Dataset config not found: {config_path}")
        return
    
    print("Discovering datasets from config...")
    datasets = discover_cases_from_config(config_path, project_root)
    
    if not datasets:
        print("❌ No datasets found via config!")
        return
    
    # Apply dataset filtering if requested
    if args.datasets:
        requested = {name.strip().lower() for name in args.datasets.split(',') if name.strip()}
        datasets = {
            name: info
            for name, info in datasets.items()
            if name.lower() in requested or info['category'].lower() in requested
        }
        if not datasets:
            print(f"❌ No datasets matched filter: {args.datasets}")
            return
    
    # Apply per-dataset case limits for faster iteration
    if args.max_cases is not None and args.max_cases > 0:
        for info in datasets.values():
            info['cases'] = info['cases'][:args.max_cases]
    
    print(f"✅ Ready to process {len(datasets)} datasets:")
    for dataset_name, info in datasets.items():
        print(f"   - {dataset_name}: {len(info['cases'])} cases")
    print()
    
    debug_case_tuple = None
    if args.debug_case:
        if ':' not in args.debug_case:
            print("❌ --debug-case must be in the format dataset:case_id")
            return
        ds_name, case_name = args.debug_case.split(':', 1)
        debug_case_tuple = (ds_name.strip(), case_name.strip())
    
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    output_csv = project_root / "results" / "sam_dice_scores_11clicks.csv"
    
    if args.min_voxels or args.min_dimension:
        print(f"\n🔍 Filtering criteria:")
        if args.min_voxels:
            print(f"   - Minimum voxels: {args.min_voxels}")
        if args.min_dimension:
            print(f"   - Minimum dimension: {args.min_dimension}")
        print()
    
    df = compute_all_dice_scores(
        model,
        datasets,
        device,
        img_size=img_size,
        num_clicks=num_clicks,
        threshold=threshold,
        output_csv=output_csv,
        verbose=args.verbose,
        debug_case=debug_case_tuple,
        debug_dir=debug_dir,
        min_voxels=args.min_voxels,
        min_dimension=args.min_dimension,
    )
    
    print(f"\n✅ Done! Results saved to: {output_csv}")
    
    return df


if __name__ == "__main__":
    main()
