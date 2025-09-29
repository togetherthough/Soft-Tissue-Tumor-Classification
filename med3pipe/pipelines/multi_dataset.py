from __future__ import annotations

"""
med3pipe.pipelines.multi_dataset

YAML-driven orchestrator to run the full pipeline across multiple datasets.

- Methods supported per dataset: TabPFN and LoCalPFN
- For each dataset block in configs/datasets.yaml, we execute Steps 1–8 via the
  existing single-dataset entrypoints and aggregate the results.

Primary entrypoint: run_multi_dataset(...)
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
    n_components_max: int = 500,
    random_state: int = 42,
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
    """Core implementation shared by YAML-driven and folder-driven runners."""
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

        # Execute the chosen method for this dataset via the unified single-dataset pipeline
        try:
            res = run_single_dataset(
                method=meth,
                dataset_root=ds_root,
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


def run_multi_dataset(
    config_path: Path | str,
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
    """
    Run the full pipeline for multiple datasets defined in a YAML config and aggregate results.

    Returns a dict with keys:
    - runs: List[MultiRunRecord]
    - summary_path: Optional[Path]
    - summary_df: pd.DataFrame
    """
    cfg_path = Path(config_path)
    cfg = _load_yaml(cfg_path)
    return _run_multi_core(
        cfg,
        method=method,
        dataset_names=dataset_names,
        outputs_base_dir=outputs_base_dir,
        sam3d_root=sam3d_root,
        model_type=model_type,
        checkpoint=checkpoint,
        device=device,
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


def run_pipeline(
    *,
    # Choose one of the following sources
    config_path: Optional[Path | str] = None,
    config: Optional[Dict[str, Any]] = None,
    datasets_dir: Optional[Path | str] = None,
    # Or call with a single dataset (converted to a one-item multi-run)
    dataset_root: Optional[Path | str] = None,
    category: Optional[str] = None,
    ct_name: Optional[str] = None,
    # Common controls
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
    # Prepare/Split/Extract defaults for single-dataset mode
    case_glob: Optional[str] = None,
    max_cases: Optional[int] = None,
    split_ratio: float = 0.8,
    seed: int = 2025,
    img_size: int = 128,
    # Labels for single-dataset mode
    sheet_csv: Optional[Path | str] = None,
    dataset_name: Optional[str] = None,
    subject_col: str = "Subject",
    label_col: str = "Diagnosis_binary",
    case_suffix: str = "_CT",
    # Summary output
    save_summary: bool = True,
    summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Unified entrypoint.

    Supports 4 modes:
    - config_path: YAML with a 'datasets' mapping
    - config: in-memory dict with 'datasets'
    - datasets_dir: discover datasets under a folder
    - dataset_root (+metadata): treat as a single-dataset multi-run
    """
    if sum(x is not None for x in [config_path, config, datasets_dir, dataset_root]) != 1:
        raise ValueError(
            "Specify exactly one of: config_path, config, datasets_dir, dataset_root"
        )

    if config_path is not None:
        cfg_path = Path(config_path)
        cfg = _load_yaml(cfg_path)
    elif config is not None:
        cfg = config
    elif datasets_dir is not None:
        # Discover
        sam3d_root = sam3d_root or find_default_sam3d_root()
        project_root = sam3d_root.parent.parent.resolve()
        base_dir = Path(datasets_dir)
        if not base_dir.is_absolute():
            cand = (project_root / base_dir).resolve()
            if cand.exists():
                base_dir = cand
        ds_map = discover_datasets_in_folder(base_dir, require_sheet_csv=True)
        cfg = {"datasets": ds_map}
    else:
        # Single dataset to one-item multi-run
        if dataset_root is None:
            raise ValueError("dataset_root must be provided for single-dataset mode")
        ds_root = Path(dataset_root)
        key = (category or ds_root.name)
        cat = key
        ct = ct_name or f"ct_{key.upper()}"
        labels_block = {
            "sheet_csv": str(sheet_csv) if sheet_csv else "sheet.csv",
            "dataset_name": dataset_name,
            "subject_col": subject_col,
            "label_col": label_col,
            "case_suffix": case_suffix,
        }
        cfg = {
            "datasets": {
                key: {
                    "dataset_root": str(ds_root),
                    "category": cat,
                    "ct_name": ct,
                    "labels": labels_block,
                    "prepare": {"case_glob": case_glob, "max_cases": max_cases},
                    "split": {"ratio": split_ratio, "seed": seed},
                    "extraction": {"img_size": img_size},
                }
            }
        }

    return _run_multi_core(
        cfg,
        method=method,
        dataset_names=dataset_names,
        outputs_base_dir=outputs_base_dir,
        sam3d_root=sam3d_root,
        model_type=model_type,
        checkpoint=checkpoint,
        device=device,
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
    **kwargs: Any,
) -> Dict[str, Any]:
    """YAML-driven runner that executes only the TabPFN method across datasets."""
    return run_multi_dataset(config_path=config_path, method="tabpfn", **kwargs)


def run_multi_localpfn(
    config_path: Path | str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """YAML-driven runner that executes only the LoCalPFN method across datasets."""
    return run_multi_dataset(config_path=config_path, method="localpfn", **kwargs)


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
