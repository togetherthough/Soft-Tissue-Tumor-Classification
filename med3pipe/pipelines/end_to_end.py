from __future__ import annotations

"""
med3pipe.pipelines.end_to_end

High-level pipeline to run Steps 1–8 (SAM-Med3D -> TabPFN/LoCalPFN) in one call.

Flow:
1-2) Prepare dataset to SAM-Med3D nnU-Net-style folders.
3)    Create validation split (copy by default) into imagesVal/labelsVal (folder-level, for caching only).
4)    Build SAM-Med3D model (optionally load checkpoint).
4)    Extract TRAIN and VAL embeddings.
5)    Pool to per-case vectors using Global Average Pooling.
6)    Load labels and build label map.
6b)   Perform a STRATIFIED feature-level split from the UNION of features (train_ratio).
7)    Standardize (fit on TRAIN) + PCA.
8)    Train the chosen tabular method and evaluate on VAL. Save all artifacts by default.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from ..data.prepare import (
    Sam3DPaths,
    find_default_sam3d_root,
    prepare_for_sam3d,
    split_validation,
)
from ..sam.core import (
    build_sam3d_model,
    default_feature_dirs,
    extract_embeddings_train_val,
    load_pooled_features,
    load_labels_from_sheet,
    build_y,
)
from ..tabular.tabpfn import (
    tabpfn_pipeline,
    default_tabpfn_out_dir,
)
from ..tabular.localpfn import (
    localpfn_pipeline,
    LocalPFNConfig,
    default_localpfn_out_dir,
)
from ..tabular.stratify import stratified_features_split


@dataclass
class EndToEndResult:
    paths: Sam3DPaths
    feature_dirs: Any
    X_train: np.ndarray
    X_val: np.ndarray
    ids_train: list[str]
    ids_val: list[str]
    y_train: np.ndarray
    y_val: np.ndarray
    tabpfn: Dict[str, Any]


@dataclass
class LocalEndToEndResult:
    paths: Sam3DPaths
    feature_dirs: Any
    X_train: np.ndarray
    X_val: np.ndarray
    ids_train: list[str]
    ids_val: list[str]
    y_train: np.ndarray
    y_val: np.ndarray
    localpfn: Dict[str, Any]


def run_single_dataset(
    method: str,
    *,
    dataset_root: Path,
    category: str = "gist",
    ct_name: str = "ct_GIST",
    # Discovery/prepare
    case_glob: Optional[str] = None,
    max_cases: Optional[int] = None,
    image_pattern: Optional[str] = None,
    seg_pattern: Optional[str] = None,
    # Split
    split_ratio: float = 0.8,
    seed: int = 2025,
    # SAM3D model/extraction
    sam3d_root: Optional[Path] = None,
    model_type: str = "vit_b_ori",
    checkpoint: Optional[Path] = None,
    img_size: int = 128,
    device: Optional[str] = None,
    skip_existing_embeddings: bool = True,
    # Labels
    sheet_csv: Optional[Path] = None,
    dataset_name: Optional[str] = "GIST",
    subject_col: str = "Subject",
    label_col: str = "Diagnosis_binary",
    case_suffix: str = "_CT",
    # Shared tabular params
    n_components_max: int = 500,
    random_state: int = 42,
    # TabPFN-specific
    tabpfn_out_dir: Optional[Path] = None,
    tabpfn_src: Optional[Path] = None,
    clf_kwargs: Optional[Dict[str, Any]] = None,
    # LoCalPFN-specific
    local_out_dir: Optional[Path] = None,
    local_cfg: Optional[LocalPFNConfig] = None,
) -> EndToEndResult | LocalEndToEndResult:
    """Unified single-dataset pipeline.

    Runs Steps 1–6 (prepare, split, extract, pool, labels) once, then branches to
    TabPFN or LoCalPFN for Steps 7–8 depending on `method`.
    """
    dataset_root = Path(dataset_root)
    sam3d_root = sam3d_root or find_default_sam3d_root()
    # Anchor relative dataset_root to project root if possible
    if not dataset_root.is_absolute() and not dataset_root.exists():
        proj_root = sam3d_root.parent.parent  # <PROJECT_ROOT>
        anchored = proj_root / dataset_root
        if anchored.exists():
            dataset_root = anchored

    # 1–2) Prepare into imagesTr/labelsTr
    prepared, paths = prepare_for_sam3d(
        dataset_root=dataset_root,
        sam3d_root=sam3d_root,
        category=category,
        ct_name=ct_name,
        case_glob=case_glob,
        max_cases=max_cases,
        image_pattern=image_pattern,
        seg_pattern=seg_pattern,
    )

    # 3) Split into imagesVal/labelsVal (copy by default)
    split_validation(paths, split_ratio=split_ratio, seed=seed, copy=True)

    # 4) Build model and extract embeddings
    torch_device = None
    if device is not None:
        torch_device = torch.device(device)
    model = build_sam3d_model(
        sam3d_root=sam3d_root,
        model_type=model_type,
        checkpoint=checkpoint,
        device=torch_device,
        eval_mode=True,
    )
    feat_dirs = default_feature_dirs(sam3d_root, category=category, ct_name=ct_name)
    extract_embeddings_train_val(
        paths,
        model,
        sam3d_root=sam3d_root,
        img_size=img_size,
        feature_dirs=feat_dirs,
        device=torch_device,
        skip_existing=skip_existing_embeddings,
    )

    # 6) Labels -> build map
    if sheet_csv is None:
        candidates = [
            dataset_root / "sheet.csv",
            Path.cwd() / category / "sheet.csv",
            sam3d_root.parent.parent / category / "sheet.csv",
        ]
        sheet_csv = next((c for c in candidates if c.exists()), None)
        if sheet_csv is None:
            raise FileNotFoundError(
                f"sheet.csv not found. Tried: {[str(c) for c in candidates]}. Pass sheet_csv or correct dataset_root."
            )
    df, lab_map = load_labels_from_sheet(
        sheet_csv=sheet_csv,
        dataset_name=dataset_name,
        subject_col=subject_col,
        label_col=label_col,
        case_suffix=case_suffix,
    )

    # 6b) Stratified feature-level split from the UNION of features
    (X_train, y_train, ids_train), (X_val, y_val, ids_val) = stratified_features_split(
        feat_train_dir=feat_dirs.train_dir,
        feat_val_dir=feat_dirs.val_dir,
        labels_tr_dir=paths.labels_tr,
        labels_val_dir=paths.labels_val,
        lab_map=lab_map,
        train_ratio=split_ratio,
        seed=seed,
    )

    method_l = method.lower()
    if method_l == "tabpfn":
        out_dir = tabpfn_out_dir or default_tabpfn_out_dir(category, ct_name)
        tabpfn_res = tabpfn_pipeline(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            ids_val=ids_val,
            category=category,
            ct_name=ct_name,
            out_dir=out_dir,
            n_components_max=n_components_max,
            random_state=random_state,
            device=None,  # auto
            tabpfn_src=tabpfn_src,
            clf_kwargs=clf_kwargs,
        )
        return EndToEndResult(
            paths=paths,
            feature_dirs=feat_dirs,
            X_train=X_train,
            X_val=X_val,
            ids_train=ids_train,
            ids_val=ids_val,
            y_train=y_train,
            y_val=y_val,
            tabpfn=tabpfn_res,
        )
    elif method_l == "localpfn":
        out_dir = local_out_dir or default_localpfn_out_dir(category, ct_name)
        local_res = localpfn_pipeline(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            ids_val=ids_val,
            category=category,
            ct_name=ct_name,
            out_dir=out_dir,
            n_components_max=n_components_max,
            random_state=random_state,
            cfg=local_cfg,
        )
        return LocalEndToEndResult(
            paths=paths,
            feature_dirs=feat_dirs,
            X_train=X_train,
            X_val=X_val,
            ids_train=ids_train,
            ids_val=ids_val,
            y_train=y_train,
            y_val=y_val,
            localpfn=local_res,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def run_end_to_end(
    dataset_root: Path,
    category: str = "gist",
    ct_name: str = "ct_GIST",
    # Discovery/prepare
    case_glob: Optional[str] = None,
    max_cases: Optional[int] = None,
    # Split
    split_ratio: float = 0.8,
    seed: int = 2025,
    # SAM3D model/extraction
    sam3d_root: Optional[Path] = None,
    model_type: str = "vit_b_ori",
    checkpoint: Optional[Path] = None,
    img_size: int = 128,
    device: Optional[str] = None,  # 'cuda' or 'cpu'; if None, auto
    # Labels
    sheet_csv: Optional[Path] = None,
    dataset_name: Optional[str] = "GIST",
    subject_col: str = "Subject",
    label_col: str = "Diagnosis_binary",
    case_suffix: str = "_CT",
    # TabPFN
    n_components_max: int = 500,
    random_state: int = 42,
    tabpfn_out_dir: Optional[Path] = None,
    tabpfn_src: Optional[Path] = None,
    clf_kwargs: Optional[Dict[str, Any]] = None,
) -> EndToEndResult:
    """Wrapper for backward-compatibility. Delegates to `run_single_dataset(method='tabpfn', ...)`."""
    return run_single_dataset(
        method="tabpfn",
        dataset_root=dataset_root,
        category=category,
        ct_name=ct_name,
        case_glob=case_glob,
        max_cases=max_cases,
        split_ratio=split_ratio,
        seed=seed,
        sam3d_root=sam3d_root,
        model_type=model_type,
        checkpoint=checkpoint,
        img_size=img_size,
        device=device,
        sheet_csv=sheet_csv,
        dataset_name=dataset_name,
        subject_col=subject_col,
        label_col=label_col,
        case_suffix=case_suffix,
        n_components_max=n_components_max,
        random_state=random_state,
        tabpfn_out_dir=tabpfn_out_dir,
        tabpfn_src=tabpfn_src,
        clf_kwargs=clf_kwargs,
    )


def local_end_to_end(
    dataset_root: Path,
    category: str = "gist",
    ct_name: str = "ct_GIST",
    # Discovery/prepare
    case_glob: Optional[str] = None,
    max_cases: Optional[int] = None,
    # Split
    split_ratio: float = 0.8,
    seed: int = 2025,
    # SAM3D model/extraction
    sam3d_root: Optional[Path] = None,
    model_type: str = "vit_b_ori",
    checkpoint: Optional[Path] = None,
    img_size: int = 128,
    device: Optional[str] = None,  # 'cuda' or 'cpu'; if None, auto
    # Labels
    sheet_csv: Optional[Path] = None,
    dataset_name: Optional[str] = "GIST",
    subject_col: str = "Subject",
    label_col: str = "Diagnosis_binary",
    case_suffix: str = "_CT",
    # LoCalPFN
    n_components_max: int = 500,
    random_state: int = 42,
    local_out_dir: Optional[Path] = None,
    local_cfg: Optional[LocalPFNConfig] = None,
) -> LocalEndToEndResult:
    """Wrapper for backward-compatibility. Delegates to `run_single_dataset(method='localpfn', ...)`."""
    return run_single_dataset(
        method="localpfn",
        dataset_root=dataset_root,
        category=category,
        ct_name=ct_name,
        case_glob=case_glob,
        max_cases=max_cases,
        split_ratio=split_ratio,
        seed=seed,
        sam3d_root=sam3d_root,
        model_type=model_type,
        checkpoint=checkpoint,
        img_size=img_size,
        device=device,
        sheet_csv=sheet_csv,
        dataset_name=dataset_name,
        subject_col=subject_col,
        label_col=label_col,
        case_suffix=case_suffix,
        n_components_max=n_components_max,
        random_state=random_state,
        local_out_dir=local_out_dir,
        local_cfg=local_cfg,
    )


def local_from_prepared_to_localpfn(
    paths: Sam3DPaths,
    # SAM3D model/extraction
    sam3d_root: Optional[Path] = None,
    model_type: str = "vit_b_ori",
    checkpoint: Optional[Path] = None,
    img_size: int = 128,
    device: Optional[str] = None,  # 'cuda' or 'cpu'; if None, auto
    # Labels
    sheet_csv: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
    dataset_name: Optional[str] = "GIST",
    subject_col: str = "Subject",
    label_col: str = "Diagnosis_binary",
    case_suffix: str = "_CT",
    # LoCalPFN
    n_components_max: int = 500,
    random_state: int = 42,
    local_out_dir: Optional[Path] = None,
    local_cfg: Optional[LocalPFNConfig] = None,
) -> LocalEndToEndResult:
    """Backward-compatible wrapper delegating to shared `run_from_prepared` for method 'localpfn'."""
    return run_from_prepared(
        paths=paths,
        method="localpfn",
        sam3d_root=sam3d_root,
        model_type=model_type,
        checkpoint=checkpoint,
        img_size=img_size,
        device=device,
        sheet_csv=sheet_csv,
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        subject_col=subject_col,
        label_col=label_col,
        case_suffix=case_suffix,
        n_components_max=n_components_max,
        random_state=random_state,
        local_out_dir=local_out_dir,
        local_cfg=local_cfg,
    )


def run_from_prepared_to_tabpfn(
    paths: Sam3DPaths,
    # SAM3D model/extraction
    sam3d_root: Optional[Path] = None,
    model_type: str = "vit_b_ori",
    checkpoint: Optional[Path] = None,
    img_size: int = 128,
    device: Optional[str] = None,  # 'cuda' or 'cpu'; if None, auto
    # Labels
    sheet_csv: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
    dataset_name: Optional[str] = "GIST",
    subject_col: str = "Subject",
    label_col: str = "Diagnosis_binary",
    case_suffix: str = "_CT",
    # TabPFN
    n_components_max: int = 500,
    random_state: int = 42,
    tabpfn_out_dir: Optional[Path] = None,
    tabpfn_src: Optional[Path] = None,
    clf_kwargs: Optional[Dict[str, Any]] = None,
) -> EndToEndResult:
    """Backward-compatible wrapper delegating to shared `run_from_prepared` for method 'tabpfn'."""
    return run_from_prepared(
        paths=paths,
        method="tabpfn",
        sam3d_root=sam3d_root,
        model_type=model_type,
        checkpoint=checkpoint,
        img_size=img_size,
        device=device,
        sheet_csv=sheet_csv,
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        subject_col=subject_col,
        label_col=label_col,
        case_suffix=case_suffix,
        n_components_max=n_components_max,
        random_state=random_state,
        tabpfn_out_dir=tabpfn_out_dir,
        tabpfn_src=tabpfn_src,
        clf_kwargs=clf_kwargs,
    )


def run_from_prepared(
    *,
    paths: Sam3DPaths,
    method: str,
    # SAM3D model/extraction
    sam3d_root: Optional[Path] = None,
    model_type: str = "vit_b_ori",
    checkpoint: Optional[Path] = None,
    img_size: int = 128,
    device: Optional[str] = None,
    # Labels
    sheet_csv: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
    dataset_name: Optional[str] = "GIST",
    subject_col: str = "Subject",
    label_col: str = "Diagnosis_binary",
    case_suffix: str = "_CT",
    # Shared tabular params
    n_components_max: int = 500,
    random_state: int = 42,
    # TabPFN-specific
    tabpfn_out_dir: Optional[Path] = None,
    tabpfn_src: Optional[Path] = None,
    clf_kwargs: Optional[Dict[str, Any]] = None,
    # LoCalPFN-specific
    local_out_dir: Optional[Path] = None,
    local_cfg: Optional[LocalPFNConfig] = None,
) -> EndToEndResult | LocalEndToEndResult:
    """Run Steps 4–8 from prepared paths using either TabPFN or LoCalPFN."""
    sam3d_root = sam3d_root or find_default_sam3d_root()

    torch_device = None
    if device is not None:
        torch_device = torch.device(device)
    model = build_sam3d_model(
        sam3d_root=sam3d_root,
        model_type=model_type,
        checkpoint=checkpoint,
        device=torch_device,
        eval_mode=True,
    )
    feat_dirs = default_feature_dirs(sam3d_root, category=paths.category, ct_name=paths.ct_name)
    extract_embeddings_train_val(
        paths,
        model,
        sam3d_root=sam3d_root,
        img_size=img_size,
        feature_dirs=feat_dirs,
        device=torch_device,
    )

    # Labels
    if sheet_csv is None:
        if dataset_root is not None:
            sheet_csv = Path(dataset_root) / "sheet.csv"
        else:
            candidates = [
                Path.cwd() / paths.category / "sheet.csv",
                sam3d_root.parent.parent / paths.category / "sheet.csv",
            ]
            sheet_csv = next((c for c in candidates if c.exists()), None)
            if sheet_csv is None:
                raise FileNotFoundError("sheet.csv not found; pass sheet_csv or dataset_root explicitly")

    df, lab_map = load_labels_from_sheet(
        sheet_csv=sheet_csv,
        dataset_name=dataset_name,
        subject_col=subject_col,
        label_col=label_col,
        case_suffix=case_suffix,
    )

    # Stratified feature-level split from the UNION of features
    # Use default controls here (run_from_prepared does not expose split controls)
    train_ratio = 0.8
    seed = 2025
    (X_train, y_train, ids_train), (X_val, y_val, ids_val) = stratified_features_split(
        feat_train_dir=feat_dirs.train_dir,
        feat_val_dir=feat_dirs.val_dir,
        labels_tr_dir=paths.labels_tr,
        labels_val_dir=paths.labels_val,
        lab_map=lab_map,
        train_ratio=train_ratio,
        seed=seed,
    )

    method_l = method.lower()
    if method_l == "tabpfn":
        out_dir = tabpfn_out_dir or default_tabpfn_out_dir(paths.category, paths.ct_name)
        tabpfn_res = tabpfn_pipeline(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            ids_val=ids_val,
            category=paths.category,
            ct_name=paths.ct_name,
            out_dir=out_dir,
            n_components_max=n_components_max,
            random_state=random_state,
            device=None,
            tabpfn_src=tabpfn_src,
            clf_kwargs=clf_kwargs,
        )
        return EndToEndResult(
            paths=paths,
            feature_dirs=feat_dirs,
            X_train=X_train,
            X_val=X_val,
            ids_train=ids_train,
            ids_val=ids_val,
            y_train=y_train,
            y_val=y_val,
            tabpfn=tabpfn_res,
        )
    elif method_l == "localpfn":
        out_dir = local_out_dir or default_localpfn_out_dir(paths.category, paths.ct_name)
        local_res = localpfn_pipeline(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            ids_val=ids_val,
            category=paths.category,
            ct_name=paths.ct_name,
            out_dir=out_dir,
            n_components_max=n_components_max,
            random_state=random_state,
            cfg=local_cfg,
        )
        return LocalEndToEndResult(
            paths=paths,
            feature_dirs=feat_dirs,
            X_train=X_train,
            X_val=X_val,
            ids_train=ids_train,
            ids_val=ids_val,
            y_train=y_train,
            y_val=y_val,
            localpfn=local_res,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
