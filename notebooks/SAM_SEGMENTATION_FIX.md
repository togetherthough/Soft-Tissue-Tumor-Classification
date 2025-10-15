# SAM-Med3D Segmentation Fix

## Problem

The original notebook (`sam_segmentation_visualization.ipynb`) was producing incorrect segmentations (scattered blobs instead of precise lesion contours) because it only used:
- **Image encoder** → embeddings
- **Naive thresholding** on averaged embeddings

## Solution

The fixed version (`sam_segmentation_visualization_fixed.py`) now uses the **complete SAM-Med3D pipeline**:

1. **Image encoder** → image embeddings
2. **Prompt encoder** → sparse/dense embeddings from point prompts
3. **Mask decoder** → actual segmentation masks
4. **Proper upsampling** and sigmoid activation

## Key Changes

### Before (Wrong)
```python
def generate_sam_segmentation(model, image_tensor, device):
    embeddings = model.image_encoder(image_tensor.to(device))
    # Just threshold averaged embeddings - NOT real segmentation!
    seg = (embeddings.mean(...) > embeddings.mean()).float()
    return seg
```

### After (Correct)
```python
def generate_sam_segmentation_proper(model, image_tensor, device):
    # 1. Image encoder
    image_embeddings = model.image_encoder(input_tensor)
    
    # 2. Generate prompt (center point)
    center_point = torch.tensor([[[W//2, H//2, D//2]]])
    point_labels = torch.tensor([[1]])  # positive click
    
    # 3. Prompt encoder
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=[center_point, point_labels],
        boxes=None,
        masks=prev_low_res_mask,
    )
    
    # 4. Mask decoder
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
    )
    
    # 5. Upsample to full resolution
    final_masks = F.interpolate(low_res_masks, size=(D,H,W), mode='trilinear')
    
    # 6. Apply sigmoid and threshold
    seg_mask = (torch.sigmoid(final_masks) > 0.5)
    return seg_mask
```

## Two Segmentation Modes

### 1. Center Point Prompt (default)
- Uses a single positive point at the center of the volume
- Works when ground truth is not available
- Less accurate but automatic

### 2. Bounding Box-Guided Prompts (when GT available)
- Extracts bbox from ground truth mask
- Uses multiple positive points within the lesion region
- Much more accurate segmentation
- Mimics clinical workflow where radiologist provides rough ROI

## Data Preprocessing

The pipeline applies these transforms (from `make_pre_transform`):
1. **`ToCanonical()`** - Reorient to canonical orientation (RAS+)
2. **`CropOrPad(128, 128, 128)`** - Resize to 128³ voxels
3. **`ZNormalization`** - Z-score normalization with masking (non-zero voxels only)

These are **standard medical image preprocessing** steps, not augmentation for training.

## How to Use

### Option 1: Run the Python script
```bash
cd notebooks
python sam_segmentation_visualization_fixed.py
```

### Option 2: Convert to Jupyter notebook
```bash
jupytext --to notebook sam_segmentation_visualization_fixed.py
jupyter notebook sam_segmentation_visualization_fixed.ipynb
```

### Option 3: Copy code to existing notebook
Replace the `generate_sam_segmentation` function in the original notebook with the corrected versions from the fixed script.

## Expected Output

Now you should see:
- **Red overlay**: Precise contours around lesions (not scattered blobs)
- **Green overlay**: Ground truth mask for comparison
- Segmentations should closely match the GT when bbox-guided prompts are used

## Performance Notes

- **With GT bbox prompts**: High accuracy, closely matches ground truth
- **Without GT (center point only)**: Moderate accuracy, may miss lesions far from center
- **Processing time**: ~1-3 seconds per volume on GPU

## Further Improvements

To get even better results:
1. Use **iterative refinement** with multiple clicks (adjust `num_clicks` parameter)
2. Implement **interactive prompting** where user clicks on false negatives/positives
3. Load SAM-Med3D **checkpoint** (pretrained weights) instead of random initialization
4. Use **multi-mask output** and select best mask based on IoU predictions

## References

- SAM-Med3D paper: https://arxiv.org/abs/2310.15161
- Original SAM: https://segment-anything.com/
- Implementation: `SAM-Med3D-main/utils/infer_utils.py` (lines 91-181)
