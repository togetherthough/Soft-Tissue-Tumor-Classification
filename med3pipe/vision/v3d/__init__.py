"""
med3pipe.vision.v3d

3D volumetric classification training (DenseNet121 3D, Swin 3D, ViT 3D).
Re-exports from local module med3pipe.vision.v3d.vision3d.
"""

from .vision3d import (
    Train3DConfig,
    build_densenet121_3d,
    build_swin_transformer_3d,
    train_eval_densenet121_3d,
    train_eval_swin_transformer_3d,
    build_vit_3d,
    train_eval_vit_3d,
)

__all__ = [
    "Train3DConfig",
    "build_densenet121_3d",
    "build_swin_transformer_3d",
    "train_eval_densenet121_3d",
    "train_eval_swin_transformer_3d",
    "build_vit_3d",
    "train_eval_vit_3d",
]
