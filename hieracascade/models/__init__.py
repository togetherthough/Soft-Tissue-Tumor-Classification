"""Model architectures for HieraCascade-STS"""

from .swin3d import (
    SwinTransformer3D,
    build_swin3d_tiny,
    build_swin3d_small,
    build_swin3d_base,
)

from .pooling import (
    SetTransformer,
    AttentionMIL,
    GatedAttentionMIL,
    build_pooling,
)

from .stage1 import (
    Stage1Model,
    build_stage1_model,
)

from .stage2 import (
    Stage2Model,
    build_stage2_model,
)

__all__ = [
    # Backbones
    "SwinTransformer3D",
    "build_swin3d_tiny",
    "build_swin3d_small",
    "build_swin3d_base",
    # Pooling
    "SetTransformer",
    "AttentionMIL",
    "GatedAttentionMIL",
    "build_pooling",
    # Stage models
    "Stage1Model",
    "build_stage1_model",
    "Stage2Model",
    "build_stage2_model",
]
