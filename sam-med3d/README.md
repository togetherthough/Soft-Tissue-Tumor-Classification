# SAM-Med3D Resources

This directory contains SAM-Med3D model checkpoints and cached features.

## Directory Structure

```
sam-med3d/
├── ckpt/              # Model checkpoints
│   └── sam_med3d_turbo.pth
├── features/          # Cached embedding features
└── README.md          # This file
```

## Checkpoints

### `sam_med3d_turbo.pth`
- **Source**: [HuggingFace - blueyo0/SAM-Med3D](https://huggingface.co/blueyo0/SAM-Med3D)
- **Size**: ~383 MB
- **Purpose**: Pre-trained SAM-Med3D encoder for medical image feature extraction
- **Architecture**: Vision Transformer-B (ViT-B) adapted for 3D medical imaging

The checkpoint is automatically downloaded by the codebase if not present.

## Features Cache

The `features/` directory stores extracted embeddings to avoid re-computing them:

```
features/
├── gist/
│   ├── gist_train.npy       # Training embeddings
│   └── gist_validation.npy  # Validation embeddings
├── lipo/
└── ...
```

**Format**: Each `.npy` file contains a dictionary with:
- `embeddings`: (N, C, D, H, W) tensor of image encoder outputs
- `case_ids`: List of corresponding case identifiers
- `metadata`: Additional information (labels, paths, etc.)

## Loading the Model

The codebase uses the `medim` library to load SAM-Med3D:

```python
import medim
from pathlib import Path

checkpoint_path = Path("sam-med3d/ckpt/sam_med3d_turbo.pth")

model = medim.create_model(
    "SAM-Med3D",
    pretrained=True,
    checkpoint_path=str(checkpoint_path)
)
```

## Reference

- **Original Repository**: [SAM-Med3D GitHub](https://github.com/uni-medical/SAM-Med3D)
- **Paper**: "SAM-Med3D: A Foundation Model for 3D Medical Image Segmentation"

---

**Back to main**: [../README.md](../README.md)
