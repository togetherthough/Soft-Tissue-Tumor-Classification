"""
med3pipe.data

Dataset preparation and path utilities for SAM-Med3D (steps 1–3).
This subpackage re-exports stable APIs from the legacy flat module layout.
"""

from .prepare import (
    Sam3DPaths,
    find_default_sam3d_root,
    prepare_for_sam3d,
    split_validation,
)

__all__ = [
    "Sam3DPaths",
    "find_default_sam3d_root",
    "prepare_for_sam3d",
    "split_validation",
]
