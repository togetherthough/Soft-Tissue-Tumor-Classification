"""
med3pipe.training

Training helpers such as SAM-Med3D fine-tuning wrappers.
"""

from .finetune import (
    finetune_sam3d,
    patch_sam3d_data_paths,
)

__all__ = [
    "finetune_sam3d",
    "patch_sam3d_data_paths",
]
