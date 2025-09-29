from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

"""
CLI for med3pipe. Imports heavy dependencies lazily inside command handlers so that
`python -m med3pipe --help` works even if runtime deps like SimpleITK are not installed yet.
"""


def _add_common_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=False,
        help="Path to raw dataset root (e.g., ./gist). If omitted, defaults to ./gist if present.",
    )
    p.add_argument(
        "--case-glob",
        type=str,
        default=None,
        help="Glob relative to dataset-root to find case NIFTI dirs (e.g., 'GIST-*_CT/1/NIFTI').",
    )
    p.add_argument(
        "--category",
        type=str,
        default="gist",
        help="Category folder under SAM-Med3D data (default: gist)",
    )
    p.add_argument(
        "--ct-name",
        type=str,
        default="ct_GIST",
        help="CT subfolder name under SAM-Med3D data (default: ct_GIST)",
    )
    p.add_argument(
        "--sam3d-root",
        type=Path,
        default=None,
        help="SAM-Med3D repo root (inner repo path). If omitted, auto-detects.",
    )


def _add_split_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Fraction for training set; remainder goes to validation (default: 0.8)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for splitting (default: 2025)",
    )
    p.add_argument(
        "--move",
        action="store_true",
        help="Move files (destructive) instead of copy for validation subset.",
    )


def cmd_prepare(args: argparse.Namespace) -> None:
    from .prepare import find_default_sam3d_root, prepare_for_sam3d

    dataset_root: Optional[Path] = args.dataset_root
    if dataset_root is None:
        # default to ./gist if exists
        cand = Path.cwd() / "gist"
        if not cand.is_dir():
            raise SystemExit("--dataset-root not provided and ./gist not found")
        dataset_root = cand

    sam3d_root = args.sam3d_root or find_default_sam3d_root()

    print("Using dataset-root:", dataset_root)
    print("SAM3D root:", sam3d_root)
    print("Category:", args.category, "CT name:", args.ct_name)

    n, paths = prepare_for_sam3d(
        dataset_root=dataset_root,
        sam3d_root=sam3d_root,
        category=args.category,
        ct_name=args.ct_name,
        case_glob=args.case_glob,
        max_cases=args.max_cases,
    )
    print("Prepared cases:", n)
    print("Train folder:", paths.train_root)


def cmd_split(args: argparse.Namespace) -> None:
    from .prepare import Sam3DPaths, find_default_sam3d_root, split_validation

    sam3d_root = args.sam3d_root or find_default_sam3d_root()
    paths = Sam3DPaths(sam3d_root=sam3d_root, category=args.category, ct_name=args.ct_name)
    paths.ensure()

    ntr, nval = split_validation(
        paths=paths,
        split_ratio=args.split_ratio,
        seed=args.seed,
        copy=not args.move,
    )
    print("Split done | Train:", ntr, "Val:", nval)


def cmd_prepare_split(args: argparse.Namespace) -> None:
    # First prepare
    cmd_prepare(args)
    # Then split
    cmd_split(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="med3pipe",
        description="Prepare datasets for SAM-Med3D (steps 1–3).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Prepare imagesTr/labelsTr under SAM-Med3D.")
    _add_common_dataset_args(p_prep)
    p_prep.add_argument("--max-cases", type=int, default=None, help="Limit number of cases (debug)")
    p_prep.set_defaults(func=cmd_prepare)

    p_split = sub.add_parser("split", help="Create validation split under imagesVal/labelsVal.")
    _add_common_dataset_args(p_split)
    _add_split_args(p_split)
    p_split.set_defaults(func=cmd_split)

    p_both = sub.add_parser(
        "prepare-split",
        help="Run prepare then split (imagesTr/labelsTr then imagesVal/labelsVal).",
    )
    _add_common_dataset_args(p_both)
    _add_split_args(p_both)
    p_both.add_argument("--max-cases", type=int, default=None, help="Limit number of cases (debug)")
    p_both.set_defaults(func=cmd_prepare_split)

    # Multi-dataset orchestrator
    p_multi = sub.add_parser(
        "multi",
        help="Run multi-dataset pipeline (TabPFN/LoCalPFN) from a YAML config.",
    )
    p_multi.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config (e.g., configs/datasets.yaml)",
    )
    p_multi.add_argument(
        "--method",
        type=str,
        choices=["tabpfn", "localpfn"],
        default="tabpfn",
        help="Single method to run per dataset (tabpfn or localpfn)",
    )
    p_multi.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated subset of dataset keys to run (e.g., 'gist,lipo')",
    )
    p_multi.add_argument(
        "--outputs-base",
        type=Path,
        default=None,
        help="Optional base directory to write outputs (runs will be timestamped subfolders)",
    )
    # Shared optional overrides
    p_multi.add_argument("--sam3d-root", type=Path, default=None, help="Override SAM-Med3D repo root")
    p_multi.add_argument("--model-type", type=str, default="vit_b_ori")
    p_multi.add_argument("--checkpoint", type=Path, default=None)
    p_multi.add_argument("--device", type=str, default=None, help="Force device 'cuda' or 'cpu'")
    p_multi.add_argument("--n-components-max", type=int, default=500)
    p_multi.add_argument("--random-state", type=int, default=42)
    # LoCalPFN overrides (optional)
    p_multi.add_argument("--local-k", type=int, default=None, help="Override k for local retrieval")
    p_multi.add_argument("--local-metric", type=str, default="euclidean")
    p_multi.add_argument("--local-fit-adapter", action="store_true", help="Enable adapter fine-tuning")
    p_multi.add_argument("--local-adapter-epochs", type=int, default=10)
    p_multi.add_argument("--local-adapter-lr", type=float, default=5e-2)
    p_multi.add_argument("--local-adapter-weight-decay", type=float, default=0.0)
    p_multi.add_argument("--local-adapter-num-queries", type=int, default=1000)
    p_multi.set_defaults(func=cmd_multi)

    return p


def main(argv: Optional[list[str]] = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    args.func(args)


def cmd_multi(args: argparse.Namespace) -> None:
    """Run multi-dataset pipeline from YAML.

    Heavy imports are local to avoid importing torch/itk when only parsing --help.
    """
    from .pipelines import run_multi_dataset
    from .tabular.localpfn import LocalPFNConfig

    # Backward-compat shim: if old --methods is present, take the first
    method = getattr(args, "method", None)
    if method is None and hasattr(args, "methods"):
        parts = [m.strip() for m in str(args.methods).split(",") if m.strip()]
        method = parts[0] if parts else "tabpfn"
        print(f"[WARN] --methods is deprecated; using the first entry -> --method {method}")
    if method is None:
        method = "tabpfn"
    dataset_filter = None
    if args.datasets:
        dataset_filter = [d.strip() for d in str(args.datasets).split(",") if d.strip()]

    local_cfg = LocalPFNConfig(
        k=args.local_k,
        metric=args.local_metric,
        fit_adapter=bool(args.local_fit_adapter),
        adapter_epochs=int(args.local_adapter_epochs),
        adapter_lr=float(args.local_adapter_lr),
        adapter_weight_decay=float(args.local_adapter_weight_decay),
        adapter_num_queries=int(args.local_adapter_num_queries),
    )

    res = run_multi_dataset(
        config_path=args.config,
        method=method,
        dataset_names=dataset_filter,
        outputs_base_dir=args.outputs_base,
        sam3d_root=args.sam3d_root,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=args.device,
        n_components_max=int(args.n_components_max),
        random_state=int(args.random_state),
        local_cfg=local_cfg,
        local_k=args.local_k,
        local_metric=args.local_metric,
        local_fit_adapter=bool(args.local_fit_adapter),
        local_adapter_epochs=int(args.local_adapter_epochs),
        local_adapter_lr=float(args.local_adapter_lr),
        local_adapter_weight_decay=float(args.local_adapter_weight_decay),
        local_adapter_num_queries=int(args.local_adapter_num_queries),
        save_summary=True,
        summary_path=None,
    )

    print("\nMulti-dataset run completed. Summary:")
    df = res.get("summary_df")
    try:
        # Avoid huge output; print limited rows
        print(df.to_string(max_rows=50))
    except Exception:
        print(str(df))
    if res.get("summary_path"):
        print("\nSummary CSV:", res["summary_path"]) 


if __name__ == "__main__":
    main()
