"""med3pipe: Utilities to prepare datasets for SAM-Med3D (steps 1–3)

- Discover cases in a dataset laid out like `gist/`
- Prepare nnU-Net-style folders for SAM-Med3D: imagesTr/labelsTr under data/train/<category>/<ct_NAME>
- Create a validation split by copying or moving a subset into imagesVal/labelsVal

CLI available via: `python -m med3pipe ...`
"""

from .prepare import (
    Sam3DPaths,
    find_default_sam3d_root,
    prepare_for_sam3d,
    split_validation,
)
from .sam3d import (
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
from .finetune import (
    finetune_sam3d,
)
from .tabpfn import (
    standardize_pca,
    train_eval_tabpfn,
    tabpfn_pipeline,
    default_tabpfn_out_dir,
)
from .pipeline import (
    run_end_to_end,
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
    # End-to-end (Steps 1–8)
    "run_end_to_end",
]

__version__ = "0.2.0"
