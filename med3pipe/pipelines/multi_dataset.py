from __future__ import annotations

"""
med3pipe.pipelines.multi_dataset

YAML-driven orchestrator to run the full pipeline across multiple datasets.

- Methods supported per dataset: TabPFN and LoCalPFN
- For each dataset block in configs/datasets.yaml, we execute Steps 1–8 via the
  existing single-dataset entrypoints and aggregate the results.

Method-specific entrypoints:
- run_multi_tabpfn(config_path, ...)
- run_multi_localpfn(config_path, ...)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import traceback

import pandas as pd
import yaml

from .end_to_end import run_single_dataset
from ..data.prepare import find_default_sam3d_root
from ..tabular.localpfn import LocalPFNConfig
from ..tabular.tabpfn import default_tabpfn_out_dir
from ..tabular.localpfn import default_localpfn_out_dir
from ..tabular.lesion_filter import LesionSizeFilter


@dataclass
class MultiRunRecord:
    dataset_key: str
    category: str
    ct_name: str
    method: str  # "tabpfn" | "localpfn"
    accuracy: Optional[float]
    macro_f1: Optional[float]
    roc_auc: Optional[float]
    out_dir: Optional[Path]
    metrics_path: Optional[Path]
    pred_path: Optional[Path]
    error: Optional[str] = None


def _resolve_dataset_root(
    dataset_root: Optional[str | Path],
    category: str,
    project_root: Path,
) -> Path:
    """
    Resolve dataset_root following the documented strategy:
    1) If absolute and exists, use it.
    2) Else try <PROJECT_ROOT>/<dataset_root> (if provided).
    3) Else try <PROJECT_ROOT>/<category>.
    4) Else try <PROJECT_ROOT>/data/<category>.
    """
    if dataset_root is not None:
        p = Path(dataset_root)
        if p.is_absolute() and p.exists():
            return p
        cand = (project_root / p).resolve()
        if cand.exists():
            return cand
    # Fallbacks
    c1 = (project_root / category).resolve()
    if c1.exists():
        return c1
    c2 = (project_root / "data" / category).resolve()
    if c2.exists():
        return c2
    raise FileNotFoundError(
        f"Could not resolve dataset_root. Tried: {dataset_root!r}, {c1}, {c2}"
    )


def _resolve_sheet_csv(sheet_csv: Optional[str | Path], dataset_root: Path) -> Optional[Path]:
    if sheet_csv is None:
        # Try default relative name under dataset_root
        cand = dataset_root / "sheet.csv"
        return cand if cand.exists() else None
    p = Path(sheet_csv)
    return p if p.is_absolute() else (dataset_root / p)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_multi_core(
    cfg: Dict[str, Any],
    *,
    method: str = "tabpfn",
    dataset_names: Optional[Sequence[str]] = None,
    outputs_base_dir: Optional[Path] = None,
    sam3d_root: Optional[Path] = None,
    model_type: str = "vit_b_ori",
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
    skip_existing_embeddings: bool = False,
    # ROI cropping parameters
    use_roi_crop: bool = False,
    roi_margin: int = 10,
    roi_target_size: int = 128,
    # Lesion filtering parameters
    lesion_filter: Optional[LesionSizeFilter] = None,
    min_voxels: Optional[int] = None,
    min_dimension: Optional[int] = None,
    min_density: Optional[float] = None,
    # Tabular params
    n_components_max: int = 500,
    random_state: int = 42,
    n_splits: int = 5,
    tabpfn_src: Optional[Path] = None,
    # TabPFN ablations
    tabpfn_clf_kwargs: Optional[Dict[str, Any]] = None,
    # LoCalPFN ablations
    local_cfg: Optional[LocalPFNConfig] = None,
    local_k: Optional[int] = None,
    local_metric: str = "euclidean",
    local_fit_adapter: bool = False,
    local_adapter_epochs: int = 10,
    local_adapter_lr: float = 5e-2,
    local_adapter_weight_decay: float = 0.0,
    local_adapter_num_queries: int = 1000,
    save_summary: bool = True,
    summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Core implementation shared by YAML-driven and folder-driven runners.
    
    IMPORTANT: 
    - skip_existing_embeddings=False (default): Reuse existing embeddings, extract only if missing
    - skip_existing_embeddings=True: Ignore existing embeddings and always re-extract
    
    Each dataset is checked individually, ensuring correct processing even when run
    separately or in different orders (e.g., GIST first, then LIPO).
    """
    if "datasets" not in cfg or not isinstance(cfg["datasets"], dict):
        raise ValueError("Config must contain a 'datasets:' mapping")

    sam3d_root = sam3d_root or find_default_sam3d_root()
    project_root = sam3d_root.parent.parent.resolve()

    # Determine which datasets to run
    all_ds_keys = list(cfg["datasets"].keys())
    if dataset_names:
        ds_keys = [k for k in dataset_names if k in cfg["datasets"]]
    else:
        ds_keys = all_ds_keys

    # Prepare output summary path default
    if save_summary and summary_path is None:
        summary_path = (Path.cwd() / "notebooks" / "multi_results_summary.csv").resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate method
    meth = str(method).lower().strip()
    if meth not in {"tabpfn", "localpfn"}:
        raise ValueError(f"method must be one of 'tabpfn' or 'localpfn'; got {method!r}")

    # Prepare LoCalPFN config (override or build)
    if meth == "localpfn":
        if local_cfg is None:
            local_cfg = LocalPFNConfig(
                k=local_k,
                metric=local_metric,
                fit_adapter=bool(local_fit_adapter),
                adapter_epochs=int(local_adapter_epochs),
                adapter_lr=float(local_adapter_lr),
                adapter_weight_decay=float(local_adapter_weight_decay),
                adapter_num_queries=int(local_adapter_num_queries),
            )
        else:
            # Apply overrides if provided
            if local_k is not None:
                local_cfg.k = local_k
            if local_metric is not None:
                local_cfg.metric = local_metric
            local_cfg.fit_adapter = bool(local_fit_adapter) if local_fit_adapter is not None else local_cfg.fit_adapter
            local_cfg.adapter_epochs = int(local_adapter_epochs)
            local_cfg.adapter_lr = float(local_adapter_lr)
            local_cfg.adapter_weight_decay = float(local_adapter_weight_decay)
            local_cfg.adapter_num_queries = int(local_adapter_num_queries)

    records: List[MultiRunRecord] = []

    for ds_key in ds_keys:
        ds_cfg: Dict[str, Any] = cfg["datasets"][ds_key] or {}
        category = ds_cfg.get("category", ds_key)
        ct_name = ds_cfg.get("ct_name", f"ct_{ds_key.upper()}")

        # Resolve dataset root and sheet.csv
        ds_root_raw = ds_cfg.get("dataset_root")
        ds_root = _resolve_dataset_root(ds_root_raw, category=category, project_root=project_root)

        labels = ds_cfg.get("labels", {}) or {}
        sheet_csv = _resolve_sheet_csv(labels.get("sheet_csv"), ds_root)
        dataset_name = labels.get("dataset_name")
        subject_col = labels.get("subject_col", "Subject")
        label_col = labels.get("label_col", "Diagnosis_binary")
        case_suffix = labels.get("case_suffix", "_CT")

        prep = ds_cfg.get("prepare", {}) or {}
        case_glob = prep.get("case_glob")
        max_cases = prep.get("max_cases")
        image_pattern = prep.get("image_pattern")
        seg_pattern = prep.get("seg_pattern")

        split = ds_cfg.get("split", {}) or {}
        split_ratio = float(split.get("ratio", 0.8))
        seed = int(split.get("seed", 2025))

        extract = ds_cfg.get("extraction", {}) or {}
        img_size = int(extract.get("img_size", 128))

        # Compute method-specific output dirs using provided base (if any)
        def _tabpfn_out_dir() -> Optional[Path]:
            if outputs_base_dir is None:
                return None
            return default_tabpfn_out_dir(category, ct_name, base_dir=outputs_base_dir)

        def _local_out_dir() -> Optional[Path]:
            if outputs_base_dir is None:
                return None
            return default_localpfn_out_dir(category, ct_name, base_dir=outputs_base_dir)

        # Check if embeddings exist for this specific dataset
        feat_train_dir = sam3d_root / "features" / category / f"{ct_name}_train"
        feat_val_dir = sam3d_root / "features" / category / ct_name
        
        embeddings_exist = (
            feat_train_dir.exists() 
            and feat_val_dir.exists()
            and list(feat_train_dir.glob("*_embedding.pt"))
            and list(feat_val_dir.glob("*_embedding.pt"))
        )
        
        # Determine extraction behavior:
        # skip_existing_embeddings=True → ignore existing, always extract
        # skip_existing_embeddings=False → use existing if available, only extract if missing
        if skip_existing_embeddings:
            # User wants to skip/ignore existing embeddings → force re-extraction
            skip_for_this_dataset = False
            if embeddings_exist:
                print(f"\n🔄 {ds_key}: Skipping existing embeddings. Re-extracting...")
            else:
                print(f"\n⚠️  {ds_key}: No existing embeddings. Extracting...")
        else:
            # User wants to use existing embeddings if available
            if embeddings_exist:
                skip_for_this_dataset = True
                print(f"\n✅ {ds_key}: Reusing existing embeddings...")
            else:
                skip_for_this_dataset = False
                print(f"\n⚠️  {ds_key}: Embeddings missing. Extracting...")
        
        # Execute the chosen method for this dataset via the unified single-dataset pipeline
        try:
            res = run_single_dataset(
                method=meth,
                dataset_root=ds_root,
                category=category,
                ct_name=ct_name,
                case_glob=case_glob,
                max_cases=max_cases,
                image_pattern=image_pattern,
                seg_pattern=seg_pattern,
                use_roi_crop=use_roi_crop,
                roi_margin=roi_margin,
                roi_target_size=roi_target_size,
                lesion_filter=lesion_filter,
                min_voxels=min_voxels,
                min_dimension=min_dimension,
                min_density=min_density,
                split_ratio=split_ratio,
                seed=seed,
                n_splits=n_splits,
                sam3d_root=sam3d_root,
                model_type=model_type,
                checkpoint=checkpoint,
                img_size=img_size,
                device=device,
                skip_existing_embeddings=skip_for_this_dataset,
                sheet_csv=sheet_csv,
                dataset_name=dataset_name,
                subject_col=subject_col,
                label_col=label_col,
                case_suffix=case_suffix,
                n_components_max=n_components_max,
                random_state=random_state,
                tabpfn_out_dir=_tabpfn_out_dir(),
                tabpfn_src=tabpfn_src,
                clf_kwargs=tabpfn_clf_kwargs,
                local_out_dir=_local_out_dir(),
                local_cfg=local_cfg,
            )
            if meth == "tabpfn":
                met = res.tabpfn.get("metrics", {})
                records.append(
                    MultiRunRecord(
                        dataset_key=ds_key,
                        category=category,
                        ct_name=ct_name,
                        method="tabpfn",
                        accuracy=met.get("accuracy"),
                        macro_f1=met.get("macro_f1"),
                        roc_auc=met.get("roc_auc"),
                        out_dir=res.tabpfn.get("out_dir"),
                        metrics_path=res.tabpfn.get("metrics_path"),
                        pred_path=res.tabpfn.get("pred_path"),
                    )
                )
            elif meth == "localpfn":
                met = res.localpfn.get("metrics", {})
                records.append(
                    MultiRunRecord(
                        dataset_key=ds_key,
                        category=category,
                        ct_name=ct_name,
                        method="localpfn",
                        accuracy=met.get("accuracy"),
                        macro_f1=met.get("macro_f1"),
                        roc_auc=met.get("roc_auc"),
                        out_dir=res.localpfn.get("out_dir"),
                        metrics_path=res.localpfn.get("metrics_path"),
                        pred_path=res.localpfn.get("pred_path"),
                    )
                )
        except Exception as e:
            # Record failure and continue with other datasets
            tb = traceback.format_exc(limit=2)
            records.append(
                MultiRunRecord(
                    dataset_key=ds_key,
                    category=category,
                    ct_name=ct_name,
                    method=meth,
                    accuracy=None,
                    macro_f1=None,
                    roc_auc=None,
                    out_dir=None,
                    metrics_path=None,
                    pred_path=None,
                    error=f"{e}\n{tb}",
                )
            )

    # Build summary DataFrame
    rows: List[Dict[str, Any]] = []
    for r in records:
        rows.append(
            {
                "dataset": r.dataset_key,
                "category": r.category,
                "ct_name": r.ct_name,
                "method": r.method,
                "accuracy": r.accuracy,
                "macro_f1": r.macro_f1,
                "roc_auc": r.roc_auc,
                "out_dir": str(r.out_dir) if r.out_dir else None,
                "metrics_path": str(r.metrics_path) if r.metrics_path else None,
                "pred_path": str(r.pred_path) if r.pred_path else None,
                "error": r.error,
            }
        )
    summary_df = pd.DataFrame(rows)

    out_summary_path: Optional[Path] = None
    if save_summary:
        out_summary_path = summary_path
        if out_summary_path is None:
            out_summary_path = (Path.cwd() / "notebooks" / "multi_results_summary.csv").resolve()
            out_summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_summary_path, index=False)

    return {
        "runs": records,
        "summary_path": out_summary_path,
        "summary_df": summary_df,
    }




def run_multi_tabpfn_from_folder(
    datasets_dir: Path | str = "data",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Folder-driven runner that executes only the TabPFN method across datasets."""
    return run_multi_from_folder(datasets_dir=datasets_dir, method="tabpfn", **kwargs)


def run_multi_localpfn_from_folder(
    datasets_dir: Path | str = "data",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Folder-driven runner that executes only the LoCalPFN method across datasets."""
    return run_multi_from_folder(datasets_dir=datasets_dir, method="localpfn", **kwargs)


def run_multi_tabpfn(
    config_path: Path | str,
    dataset_names: Optional[Sequence[str]] = None,
    outputs_base_dir: Optional[Path] = None,
    # Shared SAM3D params
    sam3d_root: Optional[Path] = None,
    model_type: str = "vit_b_ori",
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
    # Extraction control
    skip_existing_embeddings: bool = False,
    # ROI cropping
    use_roi_crop: bool = False,
    roi_margin: int = 10,
    roi_target_size: int = 128,
    # Lesion filtering
    lesion_filter: Optional[LesionSizeFilter] = None,
    min_voxels: Optional[int] = None,
    min_dimension: Optional[int] = None,
    min_density: Optional[float] = None,
    # Shared Tabular params
    n_components_max: int = 500,
    random_state: int = 42,
    n_splits: int = 5,
    # TabPFN specific
    tabpfn_src: Optional[Path] = None,
    tabpfn_clf_kwargs: Optional[Dict[str, Any]] = None,
    # Summary output
    save_summary: bool = True,
    summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run TabPFN only across datasets defined in a YAML config.
    
    Args:
        skip_existing_embeddings: If True, skip/ignore existing embeddings and re-extract.
                                  If False (default), reuse existing embeddings when available.
                                  Missing embeddings are always extracted regardless of this setting.
        use_roi_crop: If True, uses ROI-centric cropping (tumor-centered volumes).
                     If False (default), uses full-volume resizing.
        roi_margin: Margin in voxels around lesion bounding box (only if use_roi_crop=True).
        roi_target_size: Target size for ROI-cropped volumes (only if use_roi_crop=True).
    
    Embedding behavior:
        - skip_existing_embeddings=False (recommended): Reuse embeddings if they exist, extract if missing
        - skip_existing_embeddings=True: Always re-extract, ignoring existing embeddings
    
    The pipeline automatically checks each dataset individually, so you can safely run
    datasets in any order without worrying about missing embeddings.
    """
    cfg_path = Path(config_path)
    cfg = _load_yaml(cfg_path)
    return _run_multi_core(
        cfg,
        method="tabpfn",
        dataset_names=dataset_names,
        outputs_base_dir=outputs_base_dir,
        sam3d_root=sam3d_root,
        model_type=model_type,
        checkpoint=checkpoint,
        device=device,
        skip_existing_embeddings=skip_existing_embeddings,
        use_roi_crop=use_roi_crop,
        roi_margin=roi_margin,
        roi_target_size=roi_target_size,
        lesion_filter=lesion_filter,
        min_voxels=min_voxels,
        min_dimension=min_dimension,
        min_density=min_density,
        n_components_max=n_components_max,
        random_state=random_state,
        n_splits=n_splits,
        tabpfn_src=tabpfn_src,
        tabpfn_clf_kwargs=tabpfn_clf_kwargs,
        save_summary=save_summary,
        summary_path=summary_path,
    )


def run_multi_localpfn(
    config_path: Path | str,
    dataset_names: Optional[Sequence[str]] = None,
    outputs_base_dir: Optional[Path] = None,
    # Shared SAM3D params
    sam3d_root: Optional[Path] = None,
    model_type: str = "vit_b_ori",
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
    # Extraction control
    skip_existing_embeddings: bool = False,
    # ROI cropping
    use_roi_crop: bool = False,
    roi_margin: int = 10,
    roi_target_size: int = 128,
    # Lesion filtering  
    lesion_filter: Optional[LesionSizeFilter] = None,
    min_voxels: Optional[int] = None,
    min_dimension: Optional[int] = None,
    min_density: Optional[float] = None,
    # Shared Tabular params
    n_components_max: int = 500,
    random_state: int = 42,
    n_splits: int = 5,
    # LoCalPFN specific
    local_cfg: Optional[LocalPFNConfig] = None,
    local_k: Optional[int] = None,
    local_metric: str = "euclidean",
    local_fit_adapter: bool = False,
    local_adapter_epochs: int = 10,
    local_adapter_lr: float = 5e-2,
    local_adapter_weight_decay: float = 0.0,
    local_adapter_num_queries: int = 1000,
    # Summary output
    save_summary: bool = True,
    summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run LoCalPFN only across datasets defined in a YAML config.
    
    Args:
        skip_existing_embeddings: If True, skip/ignore existing embeddings and re-extract.
                                  If False (default), reuse existing embeddings when available.
                                  Missing embeddings are always extracted regardless of this setting.
        use_roi_crop: If True, uses ROI-centric cropping (tumor-centered volumes).
                     If False (default), uses full-volume resizing.
        roi_margin: Margin in voxels around lesion bounding box (only if use_roi_crop=True).
        roi_target_size: Target size for ROI-cropped volumes (only if use_roi_crop=True).
    
    Embedding behavior:
        - skip_existing_embeddings=False (recommended): Reuse embeddings if they exist, extract if missing
        - skip_existing_embeddings=True: Always re-extract, ignoring existing embeddings
    
    The pipeline automatically checks each dataset individually, so you can safely run
    datasets in any order without worrying about missing embeddings.
    """
    cfg_path = Path(config_path)
    cfg = _load_yaml(cfg_path)
    return _run_multi_core(
        cfg,
        method="localpfn",
        dataset_names=dataset_names,
        outputs_base_dir=outputs_base_dir,
        sam3d_root=sam3d_root,
        model_type=model_type,
        checkpoint=checkpoint,
        device=device,
        skip_existing_embeddings=skip_existing_embeddings,
        use_roi_crop=use_roi_crop,
        roi_margin=roi_margin,
        roi_target_size=roi_target_size,
        lesion_filter=lesion_filter,
        min_voxels=min_voxels,
        min_dimension=min_dimension,
        min_density=min_density,
        n_components_max=n_components_max,
        random_state=random_state,
        n_splits=n_splits,
        local_cfg=local_cfg,
        local_k=local_k,
        local_metric=local_metric,
        local_fit_adapter=local_fit_adapter,
        local_adapter_epochs=local_adapter_epochs,
        local_adapter_lr=local_adapter_lr,
        local_adapter_weight_decay=local_adapter_weight_decay,
        local_adapter_num_queries=local_adapter_num_queries,
        save_summary=save_summary,
        summary_path=summary_path,
    )


def run_multi_dataset_from_config(
    config: Dict[str, Any],
    method: str = "tabpfn",
    dataset_names: Optional[Sequence[str]] = None,
    outputs_base_dir: Optional[Path] = None,
    # Shared SAM3D params
    sam3d_root: Optional[Path] = None,
    model_type: str = "vit_b_ori",
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
    # Shared Tabular params
    n_components_max: int = 500,
    random_state: int = 42,
    # Method-specific
    tabpfn_src: Optional[Path] = None,
    # TabPFN ablations
    tabpfn_clf_kwargs: Optional[Dict[str, Any]] = None,
    # LoCalPFN ablations
    local_cfg: Optional[LocalPFNConfig] = None,
    local_k: Optional[int] = None,
    local_metric: str = "euclidean",
    local_fit_adapter: bool = False,
    local_adapter_epochs: int = 10,
    local_adapter_lr: float = 5e-2,
    local_adapter_weight_decay: float = 0.0,
    local_adapter_num_queries: int = 1000,
    # Summary output
    save_summary: bool = True,
    summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run multi-dataset using an in-memory config dict with a 'datasets' mapping."""
    return _run_multi_core(
        config,
        method=method,
        dataset_names=dataset_names,
        outputs_base_dir=outputs_base_dir,
        sam3d_root=sam3d_root,
        model_type=model_type,
        checkpoint=checkpoint,
        device=device,
        skip_existing_embeddings=skip_existing_embeddings,
        n_components_max=n_components_max,
        random_state=random_state,
        tabpfn_src=tabpfn_src,
        tabpfn_clf_kwargs=tabpfn_clf_kwargs,
        local_cfg=local_cfg,
        local_k=local_k,
        local_metric=local_metric,
        local_fit_adapter=local_fit_adapter,
        local_adapter_epochs=local_adapter_epochs,
        local_adapter_lr=local_adapter_lr,
        local_adapter_weight_decay=local_adapter_weight_decay,
        local_adapter_num_queries=local_adapter_num_queries,
        save_summary=save_summary,
        summary_path=summary_path,
    )


def discover_datasets_in_folder(
    datasets_dir: Path | str,
    *,
    require_sheet_csv: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Discover dataset subfolders under datasets_dir.

    A dataset is any immediate subdirectory. If require_sheet_csv is True, only include
    subfolders that contain a 'sheet.csv' file directly under the dataset root.

    Returns a mapping suitable for the 'datasets' section of the config.
    """
    base = Path(datasets_dir)
    if not base.is_dir():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for child in sorted([p for p in base.iterdir() if p.is_dir()]):
        key = child.name
        if require_sheet_csv and not (child / "sheet.csv").exists():
            continue
        out[key] = {
            "dataset_root": str(child),
            "category": key,
            "ct_name": f"ct_{key.upper()}",
            "labels": {
                "sheet_csv": "sheet.csv",  # relative to dataset_root
                # Leave dataset_name None by default; can be overridden later
                "dataset_name": None,
                "subject_col": "Subject",
                "label_col": "Diagnosis_binary",
                "case_suffix": "_CT",
            },
            "prepare": {
                "case_glob": None,
                "max_cases": None,
            },
            "split": {
                "ratio": 0.8,
                "seed": 2025,
            },
            "extraction": {
                "img_size": 128,
            },
        }
    return out


def run_multi_from_folder(
    datasets_dir: Path | str = "data",
    method: str = "tabpfn",
    dataset_names: Optional[Sequence[str]] = None,
    outputs_base_dir: Optional[Path] = None,
    # Shared SAM3D params
    sam3d_root: Optional[Path] = None,
    model_type: str = "vit_b_ori",
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
    # Shared Tabular params
    n_components_max: int = 500,
    random_state: int = 42,
    # Method-specific
    tabpfn_src: Optional[Path] = None,
    # TabPFN ablations
    tabpfn_clf_kwargs: Optional[Dict[str, Any]] = None,
    # LoCalPFN ablations
    local_cfg: Optional[LocalPFNConfig] = None,
    local_k: Optional[int] = None,
    local_metric: str = "euclidean",
    local_fit_adapter: bool = False,
    local_adapter_epochs: int = 10,
    local_adapter_lr: float = 5e-2,
    local_adapter_weight_decay: float = 0.0,
    local_adapter_num_queries: int = 1000,
    # Discovery params
    require_sheet_csv: bool = True,
    default_case_suffix: str = "_CT",
    # Summary output
    save_summary: bool = True,
    summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Discover datasets under a folder (default: ./data) and run TabPFN/LoCalPFN end-to-end per dataset.

    The dataset key is the subfolder name; defaults are used for label columns and case suffix
    unless overridden by editing the discovered config in-memory before running.
    """
    # Resolve project root to allow relative datasets_dir like "data"
    sam3d_root = sam3d_root or find_default_sam3d_root()
    project_root = sam3d_root.parent.parent.resolve()
    base_dir = Path(datasets_dir)
    if not base_dir.is_absolute():
        cand = (project_root / base_dir).resolve()
        if cand.exists():
            base_dir = cand

    ds_map = discover_datasets_in_folder(base_dir, require_sheet_csv=require_sheet_csv)
    # Apply default case suffix if provided
    for v in ds_map.values():
        v.setdefault("labels", {})["case_suffix"] = default_case_suffix

    cfg = {"datasets": ds_map}
    return _run_multi_core(
        cfg,
        method=method,
        dataset_names=dataset_names,
        outputs_base_dir=outputs_base_dir,
        sam3d_root=sam3d_root,
        model_type=model_type,
        checkpoint=checkpoint,
        device=device,
        skip_existing_embeddings=skip_existing_embeddings,
        n_components_max=n_components_max,
        random_state=random_state,
        tabpfn_src=tabpfn_src,
        tabpfn_clf_kwargs=tabpfn_clf_kwargs,
        local_cfg=local_cfg,
        local_k=local_k,
        local_metric=local_metric,
        local_fit_adapter=local_fit_adapter,
        local_adapter_epochs=local_adapter_epochs,
        local_adapter_lr=local_adapter_lr,
        local_adapter_weight_decay=local_adapter_weight_decay,
        local_adapter_num_queries=local_adapter_num_queries,
        save_summary=save_summary,
        summary_path=summary_path,
    )
