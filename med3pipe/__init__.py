"""med3pipe: Utilities to prepare datasets for SAM-Med3D (steps 1–3)

- Discover cases in a dataset laid out like `gist/`
- Prepare nnU-Net-style folders for SAM-Med3D: imagesTr/labelsTr under data/train/<category>/<ct_NAME>
- Create a validation split by copying or moving a subset into imagesVal/labelsVal

CLI available via: `python -m med3pipe ...`
"""

# Re-export public APIs from structured subpackages
# Ensure subpackage attribute exists (e.g., `med3pipe.vision`) when only `import med3pipe` is used
from . import vision as vision
from .data import (
    Sam3DPaths,
    find_default_sam3d_root,
    prepare_for_sam3d,
    split_validation,
)
from .sam import (
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
    load_labels_from_sheet,
    build_y,
)
from .training import (
    finetune_sam3d,
)
from .tabular import (
    standardize_pca,
    train_eval_tabpfn,
    tabpfn_pipeline,
    default_tabpfn_out_dir,
    # LoCalPFN
    LocalPFNConfig,
    localpfn_infer,
    localpfn_pipeline,
    default_localpfn_out_dir,
)
from .pipelines import (
    run_end_to_end,
    run_from_prepared_to_tabpfn,
    local_end_to_end,
    local_from_prepared_to_localpfn,
    run_multi_dataset_from_config,
    discover_datasets_in_folder,
    run_multi_from_folder,
    run_multi_tabpfn,
    run_multi_localpfn,
)
from .vision.v3d import (
    Train3DConfig,
    build_densenet121_3d,
    build_swin_transformer_3d,
    train_eval_densenet121_3d,
    train_eval_swin_transformer_3d,
    build_vit_3d,
    train_eval_vit_3d,
)

__all__ = [
    "__version__",
    # Prepare/split
    "Sam3DPaths",
    "find_default_sam3d_root",
    "prepare_for_sam3d",
    "split_validation",
    # Model/embeddings/ROI/labels
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
    "load_labels_from_sheet",
    "build_y",
    # Fine-tune
    "finetune_sam3d",
    # TabPFN (steps 7–8)
    "standardize_pca",
    "train_eval_tabpfn",
    "tabpfn_pipeline",
    "default_tabpfn_out_dir",
    # LoCalPFN
    "LocalPFNConfig",
    "localpfn_infer",
    "localpfn_pipeline",
    "default_localpfn_out_dir",
    # End-to-end (Steps 1–8)
    "run_end_to_end",
    "run_from_prepared_to_tabpfn",
    "local_end_to_end",
    "local_from_prepared_to_localpfn",
    "run_multi_dataset_from_config",
    "discover_datasets_in_folder",
    "run_multi_from_folder",
    "run_multi_tabpfn",
    "run_multi_localpfn",
    # 3D classification utilities
    "Train3DConfig",
    "build_densenet121_3d",
    "build_swin_transformer_3d",
    "train_eval_densenet121_3d",
    "train_eval_swin_transformer_3d",
    "build_vit_3d",
    "train_eval_vit_3d",
]

__version__ = "0.6.0"
