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

    return p


def main(argv: Optional[list[str]] = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
