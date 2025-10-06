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

from .stage2_binary import (
    Stage2Model as Stage2BinaryModel,
    build_stage2_model as build_stage2_binary_model,
)

from .stage2_hierarchical import (
    Stage2HierarchicalModel,
    build_stage2_hierarchical_model,
)

# For backward compatibility, Stage2Model points to binary by default
Stage2Model = Stage2BinaryModel
build_stage2_model = build_stage2_binary_model

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
    # Binary cascade (default)
    "Stage2Model",
    "build_stage2_model",
    "Stage2BinaryModel",
    "build_stage2_binary_model",
    # Hierarchical cascade  
    "Stage2HierarchicalModel",
    "build_stage2_hierarchical_model",
]
