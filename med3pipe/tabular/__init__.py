"""
med3pipe.tabular

Tabular ML utilities, including preprocessing and TabPFN training/evaluation (steps 7–8).
"""

from .tabpfn import (
    standardize_pca,
    train_eval_tabpfn,
    tabpfn_pipeline,
    default_tabpfn_out_dir,
)
from .localpfn import (
    LocalPFNConfig,
    localpfn_infer,
    localpfn_pipeline,
    default_localpfn_out_dir,
)
from .lesion_filter import (
    LesionSizeFilter,
    load_lesion_filter_from_config,
)
from .diagnostics import (
    mann_whitney_test,
    hsic_test,
    knn_agreement_test,
)

__all__ = [
    "standardize_pca",
    "train_eval_tabpfn",
    "tabpfn_pipeline",
    "default_tabpfn_out_dir",
    # LoCalPFN
    "LocalPFNConfig",
    "localpfn_infer",
    "localpfn_pipeline",
    "default_localpfn_out_dir",
    # Lesion filtering
    "LesionSizeFilter",
    "load_lesion_filter_from_config",
    # Embedding diagnostics
    "mann_whitney_test",
    "hsic_test",
    "knn_agreement_test",
]
