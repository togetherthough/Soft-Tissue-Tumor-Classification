#!/usr/bin/env python3
"""Script to update a specific cell in SAM_Visualization-2.ipynb with preprocessing strategies loop."""

import json
from pathlib import Path

# New cell content with dual preprocessing strategy loop
NEW_CELL_CODE = '''results = []
results_dir = project_root / "results" / "sam_visualization"
results_dir.mkdir(parents=True, exist_ok=True)

# ROI Cropping Helper Functions (inline)
def get_bounding_box(mask_array):
    """Get bounding box of non-zero region in 3D mask."""
    where = np.where(mask_array > 0)
    if len(where[0]) == 0:
        return None
    z_min, z_max = where[0].min(), where[0].max()
    y_min, y_max = where[1].min(), where[1].max()
    x_min, x_max = where[2].min(), where[2].max()
    return (z_min, z_max, y_min, y_max, x_min, x_max)

def add_margin_to_bbox(bbox, margin, shape):
    """Add margin to bounding box, respecting image boundaries."""
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    z_min = max(0, z_min - margin)
    z_max = min(shape[0] - 1, z_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(shape[1] - 1, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(shape[2] - 1, x_max + margin)
    return (z_min, z_max, y_min, y_max, x_min, x_max)

def crop_array_to_roi(array, bbox):
    """Crop array to ROI."""
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    return array[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

def load_with_roi_crop(img_path, mask_path, img_size, roi_margin, for_sam=True):
    """Load image and mask with ROI cropping."""
    # Load raw arrays
    sitk_img = sitk.ReadImage(str(img_path))
    sitk_arr_img, _ = tio.data.io.sitk_to_nib(sitk_img)
    
    sitk_mask = sitk.ReadImage(str(mask_path))
    sitk_arr_mask, _ = tio.data.io.sitk_to_nib(sitk_mask)
    
    # Get numpy arrays
    img_np = sitk_arr_img.squeeze()
    mask_np = sitk_arr_mask.squeeze()
    
    # Get bounding box from mask
    bbox = get_bounding_box(mask_np)
    if bbox is None:
        print("    WARNING: Empty mask, falling back to full volume")
        bbox = (0, mask_np.shape[0]-1, 0, mask_np.shape[1]-1, 0, mask_np.shape[2]-1)
    
    # Add margin
    bbox = add_margin_to_bbox(bbox, roi_margin, mask_np.shape)
    
    # Crop to ROI
    img_cropped = crop_array_to_roi(img_np, bbox)
    mask_cropped = crop_array_to_roi(mask_np, bbox)
    
    print(f"    ROI crop: {img_np.shape} -> {img_cropped.shape}")
    
    # Create TorchIO subjects from cropped arrays
    img_tensor = torch.from_numpy(img_cropped).float().unsqueeze(0)
    mask_tensor = torch.from_numpy(mask_cropped).float().unsqueeze(0)
    
    subject_img = tio.Subject(image=tio.ScalarImage(tensor=img_tensor))
    subject_mask = tio.Subject(label=tio.LabelMap(tensor=mask_tensor))
    
    # Apply CT clamping if needed
    if "/ct_" in str(img_path).lower() or "_ct" in str(img_path).lower():
        subject_img = tio.Clamp(-1000, 1000)(subject_img)
    
    # Transform: Resize + Pad (+ Normalization for SAM)
    if for_sam:
        transform = tio.Compose([
            tio.ToCanonical(),
            ResizeLargestTo(target_size=img_size),
            tio.CropOrPad(target_shape=(img_size, img_size, img_size)),
            tio.ZNormalization(masking_method=_znorm_masking_method),
        ])
    else:
        transform = tio.Compose([
            tio.ToCanonical(),
            ResizeLargestTo(target_size=img_size),
            tio.CropOrPad(target_shape=(img_size, img_size, img_size)),
        ])
    
    mask_transform = tio.Compose([
        tio.ToCanonical(),
        ResizeLargestTo(target_size=img_size),
        tio.CropOrPad(target_shape=(img_size, img_size, img_size)),
    ])
    
    subject_img = transform(subject_img)
    subject_mask = mask_transform(subject_mask)
    
    if for_sam:
        image = subject_img.image.data.clone().detach().unsqueeze(0).float()  # (1, 1, D, H, W)
    else:
        image = subject_img.image.data.squeeze().numpy()
    
    mask = subject_mask.label.data.squeeze().numpy()
    mask = (mask > 0).astype(float)
    
    return image, mask

# Run experiments with BOTH preprocessing strategies
preprocessing_strategies = [
    {"name": "ResizeLargestTo", "use_roi_crop": False, "roi_margin": 0},
    {"name": "ROI-Cropped", "use_roi_crop": True, "roi_margin": ROI_MARGIN}
]

for strategy in preprocessing_strategies:
    preproc_name = strategy["name"]
    use_roi = strategy["use_roi_crop"]
    margin = strategy["roi_margin"]
    
    print("\\n" + "="*80)
    print(f"🔬 PREPROCESSING STRATEGY: {preproc_name}")
    print("="*80)
    if use_roi:
        print(f"   • ROI-cropped with {margin}px margin")
        print(f"   • Tumor-centered volumes")
        print(f"   • Better for small lesions")
    else:
        print(f"   • Whole volume with ResizeLargestTo")
        print(f"   • Preserves anatomical context")
        print(f"   • Minimal data loss")
    print("="*80)
    
    for dataset_name, dataset_config in config['datasets'].items():
        print("\\n" + "="*60)
        print(f"Processing dataset: {dataset_name.upper()} [{preproc_name}]")
        print("="*60)
        
        images, labels = find_dataset_images(dataset_config, sam3d_root, project_root)
        print(f"Found {len(images)} images")
        
        if len(images) == 0:
            print(f"  No images found for {dataset_name}")
            continue
        
        # Only process PLOT_SAMPLES_PER_DATASET images (visualize + compute dice)
        visual_count = min(PLOT_SAMPLES_PER_DATASET, len(images)) if PLOT_SAMPLES_PER_DATASET else 0
        visual_indices = list(range(visual_count))
        
        print(f"\\n🔍 Visualizing and computing dice for {len(visual_indices)} sample(s)")
        
        for idx in visual_indices:
            img_path = images[idx]
            label_path = labels[idx] if idx < len(labels) else None
            
            print(f"\\nProcessing: {img_path.name}")
            
            try:
                # Load data with current preprocessing strategy
                if use_roi and label_path:
                    print(f"  Loading with ROI cropping ({preproc_name})...")
                    image_tensor_sam, gt_mask = load_with_roi_crop(
                        img_path, label_path, img_size, margin, for_sam=True
                    )
                    image_vol_display, _ = load_with_roi_crop(
                        img_path, label_path, img_size, margin, for_sam=False
                    )
                else:
                    print(f"  Loading image for SAM (WITH normalization)...")
                    image_tensor_sam = load_volume_for_sam(img_path, img_size=img_size)
                    
                    print(f"  Loading image for display (NO normalization)...")
                    image_vol_display = load_volume_for_display(img_path, img_size=img_size)
                    
                    print(f"  Loading ground truth mask...")
                    gt_mask = load_mask_for_sam(label_path, img_size=img_size) if label_path else None
                
                print(f"    SAM input shape: {image_tensor_sam.shape}")
                print(f"    Display shape: {image_vol_display.shape}")
                if gt_mask is not None:
                    print(f"    Mask shape: {gt_mask.shape}")
                    print(f"    Mask sum: {gt_mask.sum()} voxels")
                
                # Generate PROMPTED segmentation (now with iterative refinement!)
                print(f"\\n  🎯 Generating PROMPTED SAM segmentation (iterative refinement with {NUM_CLICKS} clicks)...")
                if gt_mask is not None:
                    seg_prompted = generate_sam_with_bbox_prompt(
                        model, image_tensor_sam, gt_mask, device
                    )
                else:
                    seg_prompted = generate_sam_with_bbox_prompt(
                        model, image_tensor_sam, np.zeros_like(image_vol_display), device
                    )
                print(f"    Prompted segmentation: {seg_prompted.sum()} voxels")
                
                # Generate UNPROMPTED segmentation
                print(f"\\n  🎯 Generating UNPROMPTED SAM segmentation (center point only)...")
                seg_unprompted = generate_sam_unprompted(
                    model, image_tensor_sam, device,
                    threshold=SEGMENTATION_THRESHOLD
                )
                print(f"    Unprompted segmentation: {seg_unprompted.sum()} voxels")
                
                # Compute Dice scores
                dice_prompted = None
                dice_unprompted = None
                if gt_mask is not None:
                    dice_prompted = compute_dice_score(seg_prompted, gt_mask)
                    dice_unprompted = compute_dice_score(seg_unprompted, gt_mask)
                    improvement = dice_prompted - dice_unprompted
                    
                    print(f"\\n  📊 DICE SCORES [{preproc_name}]:")
                    print(f"     Prompted (11-click iterative):   {dice_prompted:.4f}")
                    print(f"     Unprompted (center only): {dice_unprompted:.4f}")
                    print(f"     Improvement: {improvement:+.4f}")
                    
                    results.append({
                        "dataset": dataset_name,
                        "image": img_path.name,
                        "preprocessing": preproc_name,
                        "dice_prompted": float(dice_prompted),
                        "dice_unprompted": float(dice_unprompted),
                        "dice_improvement": float(improvement)
                    })
                    
                # Detect modality from filename
                modality = "MR" if "_MR" in img_path.name or "Lipo" in img_path.name else "CT"
                
                # Plot PROMPTED segmentation
                print(f"\\n  📸 Plotting PROMPTED segmentation [Modality: {modality}]...")
                title_prompted = f"{dataset_name.upper()}: {img_path.stem} - ITERATIVE [{preproc_name}]"
                if dice_prompted is not None:
                    title_prompted += f" (Dice: {dice_prompted:.4f})"
                
                fig = plot_orthogonal_views(
                    image_vol_display,
                    seg_prompted,
                    gt_mask,
                    title=title_prompted,
                    n_slices=3,
                    modality=modality
                )
                plt.show()
                
                # Plot UNPROMPTED segmentation
                print(f"  📸 Plotting UNPROMPTED segmentation [Modality: {modality}]...")
                title_unprompted = f"{dataset_name.upper()}: {img_path.stem} - UNPROMPTED [{preproc_name}]"
                if dice_unprompted is not None:
                    title_unprompted += f" (Dice: {dice_unprompted:.4f})"
                
                fig = plot_orthogonal_views(
                    image_vol_display,
                    seg_unprompted,
                    gt_mask,
                    title=title_unprompted,
                    n_slices=3,
                    modality=modality
                )
                plt.show()
                
            except Exception as e:
                print(f"  ❌ Error processing {img_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

# Save results
if results:
    df = pd.DataFrame(results)
    
    # Per-image results
    per_image_path = results_dir / "dice_scores_per_image.csv"
    df.to_csv(per_image_path, index=False)
    print(f"\\n💾 Saved per-image Dice scores to: {per_image_path}")
    
    # Summary statistics by preprocessing method
    summary = df.groupby(['preprocessing', 'dataset']).agg({
        'dice_prompted': ['mean', 'std', 'min', 'max'],
        'dice_unprompted': ['mean', 'std', 'min', 'max'],
        'dice_improvement': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    summary_path = results_dir / "dice_scores_summary.csv"
    summary.to_csv(summary_path)
    print(f"💾 Saved summary statistics to: {summary_path}")
    
    display(df)
    display(summary)'''

def main():
    notebook_path = Path(r"c:\Users\cahel\Desktop\Med3Tab-PFN\notebooks\visualization\SAM_Visualization-2.ipynb")
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"Loaded notebook with {len(nb['cells'])} cells")
    
    # Find the cell we want to update (look for "results = []" as first line and dataset loop)
    target_cell_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'results = []' in source and 'for dataset_name, dataset_config in config' in source:
                target_cell_idx = i
                print(f"Found target cell at index {i}")
                break
    
    if target_cell_idx is None:
        print("ERROR: Could not find target cell!")
        return
    
    # Convert new code to list of lines
    new_source = [line + '\n' for line in NEW_CELL_CODE.split('\n')]
    # Remove trailing newline from last line
    if new_source and new_source[-1] == '\n':
        new_source[-1] = ''
    
    # Update the cell
    nb['cells'][target_cell_idx]['source'] = new_source
    
    # Clear outputs
    nb['cells'][target_cell_idx]['outputs'] = []
    nb['cells'][target_cell_idx]['execution_count'] = None
    
    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Successfully updated cell {target_cell_idx}")
    print("Notebook saved!")

if __name__ == "__main__":
    main()
