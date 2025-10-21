"""
med3pipe.sam

SAM-Med3D model, embeddings, ROI pooling, and label utilities (steps 4–6).
Re-exports from the legacy flat module layout for backward compatibility.
"""

from .core import (
    Sam3DModelSpec,
    build_sam3d_model,
    make_pre_transform,
    load_volume_tensor,
    extract_embeddings,
    FeatureDirs,
    default_feature_dirs,
    extract_embeddings_train_val,
    load_mask_tensor,
    roi_pool_embedding,
    load_roi_features,
    summarize_embedding_shapes_dir,
    summarize_train_val_embedding_shapes,
    load_labels_from_sheet,
    build_y,
    ResizeLargestTo,
)

__all__ = [
    "Sam3DModelSpec",
    "build_sam3d_model",
    "make_pre_transform",
    "load_volume_tensor",
    "extract_embeddings",
    "FeatureDirs",
    "default_feature_dirs",
    "extract_embeddings_train_val",
    "load_mask_tensor",
    "roi_pool_embedding",
    "load_roi_features",
    "summarize_embedding_shapes_dir",
    "summarize_train_val_embedding_shapes",
    "load_labels_from_sheet",
    "build_y",
    "ResizeLargestTo",
]
