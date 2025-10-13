"""HieraCascade: Hierarchical Cascaded Multi-Scale 3D Transformer for Soft Tissue Tumor Classification

A mask-free two-stage architecture designed for multi-class classification:
- Stage 1: Coarse 3D Swin on full volume → saliency map + auxiliary predictions
- Stage 2: Fine 3D Swin on high-resolution crops → hierarchical classification

The model supports both CT and MRI modalities without requiring segmentation masks,
making it suitable for clinical scenarios where manual annotations are not available.
"""

__version__ = "0.1.0"

from .models import (
    Stage1Model,
    Stage2Model,
    build_stage1_model,
    build_stage2_model,
)

from .dataio import (
    DatasetStage1,
    DatasetStage2,
    SiteBalancedSampler,
    scan_data_directory,
    load_labels_csv,
    save_labels_csv,
    preprocess_volume,
)

from .losses import (
    Stage1Loss,
    Stage2Loss,
    build_stage1_loss,
    build_stage2_loss,
)

from .metrics import (
    compute_metrics,
    MetricsTracker,
)

__all__ = [
    "__version__",
    # Models
    "Stage1Model",
    "Stage2Model",
    "build_stage1_model",
    "build_stage2_model",
    # Data
    "DatasetStage1",
    "DatasetStage2",
    "SiteBalancedSampler",
    "scan_data_directory",
    "load_labels_csv",
    "save_labels_csv",
    "preprocess_volume",
    # Losses
    "Stage1Loss",
    "Stage2Loss",
    "build_stage1_loss",
    "build_stage2_loss",
    # Metrics
    "compute_metrics",
    "MetricsTracker",
]
