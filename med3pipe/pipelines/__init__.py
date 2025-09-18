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

__all__ = [
    "run_end_to_end",
    "run_from_prepared_to_tabpfn",
    "local_end_to_end",
    "local_from_prepared_to_localpfn",
    "EndToEndResult",
    "LocalEndToEndResult",
]
