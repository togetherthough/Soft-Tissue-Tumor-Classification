"""
med3pipe.pipelines

High-level orchestration pipelines that chain multiple steps end-to-end.
"""

from .end_to_end import (
    run_end_to_end,
    run_from_prepared_to_tabpfn,
    local_end_to_end,
    local_from_prepared_to_localpfn,
    EndToEndResult,
    LocalEndToEndResult,
)

from .multi_dataset import (
    run_multi_dataset_from_config,
    discover_datasets_in_folder,
    run_multi_from_folder,
    run_multi_tabpfn,
    run_multi_localpfn,
    run_multi_tabpfn_from_folder,
    run_multi_localpfn_from_folder,
)

__all__ = [
    "run_end_to_end",
    "run_from_prepared_to_tabpfn",
    "local_end_to_end",
    "local_from_prepared_to_localpfn",
    "run_multi_dataset_from_config",
    "discover_datasets_in_folder",
    "run_multi_from_folder",
    "run_multi_tabpfn",
    "run_multi_localpfn",
    "run_multi_tabpfn_from_folder",
    "run_multi_localpfn_from_folder",
    "EndToEndResult",
    "LocalEndToEndResult",
]
