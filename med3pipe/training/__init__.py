from __future__ import annotations

"""
med3pipe.training

SAM-Med3D training utilities (fine-tuning, classification head).
"""

from .finetune import finetune_sam3d, patch_sam3d_data_paths
from .classification_head import (
    run_classification_head_experiment,
    run_classification_head_experiment_kfold,
    SAMWithClassificationHead,
    TumorClassificationHead,
    train_classification_head,
    evaluate_model,
)

__all__ = [
    "finetune_sam3d",
    "patch_sam3d_data_paths",
    "run_classification_head_experiment",
    "run_classification_head_experiment_kfold",
    "SAMWithClassificationHead",
    "TumorClassificationHead",
    "train_classification_head",
    "evaluate_model",
]
