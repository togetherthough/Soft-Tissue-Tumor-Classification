from __future__ import annotations

"""
med3pipe.tabpfn

Step 7: Standardize (fit on TRAIN), then PCA to <= 500 dims; apply to VAL.
Step 8: Train TabPFN on TRAIN, evaluate on VAL, and SAVE everything by default.

This module is designed to work with features produced in steps 4–5 (ROI-pooled vectors), but you can
also use it on any numpy arrays (X_train, X_val, y_train, y_val).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import json
import sys
import time
import datetime as _dt

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import joblib


# ------------------------------
# Utilities
# ------------------------------

def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def find_default_tabpfn_src(start: Optional[Path] = None) -> Optional[Path]:
    """Search upwards for TabPFN source directory 'TabPFN-main/TabPFN-main/src'."""
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        src = cand / "TabPFN-main" / "TabPFN-main" / "src"
        if src.is_dir():
            return src
    return None


def ensure_tabpfn_on_sys_path(tabpfn_src: Optional[Path] = None) -> Optional[Path]:
    if tabpfn_src is None:
        tabpfn_src = find_default_tabpfn_src()
    if tabpfn_src is not None and str(tabpfn_src) not in sys.path:
        sys.path.insert(0, str(tabpfn_src))
    return tabpfn_src


def default_tabpfn_out_dir(
    category: str = "gist",
    ct_name: str = "ct_GIST",
    base_dir: Optional[Path] = None,
) -> Path:
    base = (base_dir or Path.cwd()).resolve()
    out = base / "tabpfn_runs" / f"{category}_{ct_name}_{_timestamp()}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ------------------------------
# Step 7: Standardize + PCA
# ------------------------------

def standardize_pca(
    X_train: np.ndarray,
    X_val: np.ndarray,
    n_components_max: int = 500,
    random_state: int = 42,
    save_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, PCA]:
    """Fit StandardScaler on TRAIN, transform both TRAIN and VAL, and fit PCA on TRAIN.

    - n_components = min(n_components_max, min(n_samples, n_features)) to avoid errors.
    - If save_dir is provided, saves scaler, pca, and transformed arrays.
    """
    assert X_train.ndim == 2 and X_val.ndim == 2, "X arrays must be 2D"

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    max_components = min(X_train_s.shape[0], X_train_s.shape[1])
    n_comp = min(n_components_max, max_components)

    pca = PCA(n_components=n_comp, svd_solver="auto", random_state=random_state)
    X_train_p = pca.fit_transform(X_train_s)
    X_val_p = pca.transform(X_val_s)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_dir / "scaler.joblib")
        joblib.dump(pca, save_dir / "pca.joblib")
        np.save(save_dir / "X_train_p.npy", X_train_p)
        np.save(save_dir / "X_val_p.npy", X_val_p)
        meta = {
            "n_components_max": n_components_max,
            "n_components_used": int(n_comp),
            "train_samples": int(X_train.shape[0]),
            "train_features": int(X_train.shape[1]),
            "val_samples": int(X_val.shape[0]),
            "val_features": int(X_val.shape[1]),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        }
        (save_dir / "preproc_meta.json").write_text(json.dumps(meta, indent=2))
    return X_train_p, X_val_p, scaler, pca


# ------------------------------
# Step 8: TabPFN Train/Eval + Save
# ------------------------------

def train_eval_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    ids_val: Sequence[str],
    out_dir: Optional[Path] = None,
    device: Optional[str] = None,
    tabpfn_src: Optional[Path] = None,
    clf_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train TabPFN on TRAIN and evaluate on VAL; save predictions and metrics.

    Saves to out_dir (created if missing):
    - tabpfn_config.json: device, classifier kwargs
    - tabpfn_val_predictions.csv: case_id, true_label, pred_label, proba_0, proba_1
    - tabpfn_metrics.json: accuracy, macro_f1, roc_auc (if proba), confusion_matrix
    - tabpfn_classification_report.txt: sklearn classification_report
    """
    # Ensure TabPFN on sys.path
    tabpfn_src = ensure_tabpfn_on_sys_path(tabpfn_src)

    # Lazy import TabPFN
    from tabpfn.classifier import TabPFNClassifier  # type: ignore

    if device is None:
        device = "cuda" if _has_cuda() else "cpu"
    clf_kwargs = clf_kwargs or {}

    clf = TabPFNClassifier(device=device, **clf_kwargs)
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = clf.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))
    macro_f1 = float(f1_score(y_val, y_pred, average="macro"))

    proba = None
    roc_auc = None
    try:
        proba = clf.predict_proba(X_val)
        if proba is not None and proba.shape[1] == 2:
            roc_auc = float(roc_auc_score(y_val, proba[:, 1]))
    except Exception:
        proba = None

    report_txt = classification_report(y_val, y_pred, target_names=["0", "1"])  # binary labels assumed
    cm = confusion_matrix(y_val, y_pred)

    # Prepare out_dir
    if out_dir is None:
        out_dir = default_tabpfn_out_dir()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    cfg = {
        "device": device,
        "clf_kwargs": clf_kwargs,
        "train_time_sec": train_time,
        "tabpfn_src": str(tabpfn_src) if tabpfn_src else None,
        "shapes": {
            "X_train": list(map(int, X_train.shape)),
            "X_val": list(map(int, X_val.shape)),
        },
    }
    (out_dir / "tabpfn_config.json").write_text(json.dumps(cfg, indent=2))

    # Save predictions
    import pandas as pd
    pred_df = {
        "case_id": list(ids_val),
        "true_label": list(map(int, y_val)),
        "pred_label": list(map(int, y_pred)),
    }
    if proba is not None:
        if proba.ndim == 1:
            proba = np.stack([1 - proba, proba], axis=-1)
        pred_df["proba_0"] = proba[:, 0].tolist()
        pred_df["proba_1"] = proba[:, 1].tolist()
    pd.DataFrame(pred_df).to_csv(out_dir / "tabpfn_val_predictions.csv", index=False)

    # Save metrics
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.astype(int).tolist(),
    }
    (out_dir / "tabpfn_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "tabpfn_classification_report.txt").write_text(report_txt)

    return {
        "clf": clf,
        "metrics": metrics,
        "pred_path": out_dir / "tabpfn_val_predictions.csv",
        "metrics_path": out_dir / "tabpfn_metrics.json",
        "config_path": out_dir / "tabpfn_config.json",
        "report_path": out_dir / "tabpfn_classification_report.txt",
        "out_dir": out_dir,
    }


def _has_cuda() -> bool:
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


# ------------------------------
# High-level pipeline (Step 7 + 8 with default saving)
# ------------------------------

def tabpfn_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    ids_val: Sequence[str],
    category: str = "gist",
    ct_name: str = "ct_GIST",
    out_dir: Optional[Path] = None,
    n_components_max: int = 500,
    random_state: int = 42,
    device: Optional[str] = None,
    tabpfn_src: Optional[Path] = None,
    clf_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Step 7 (Standardize+PCA) and Step 8 (TabPFN train/eval) and save all artifacts.

    - Creates out_dir if not provided using `tabpfn_runs/<category>_<ct_name>_<timestamp>`.
    - Saves scaler.joblib, pca.joblib, X_train_p.npy, X_val_p.npy, and preproc_meta.json under out_dir/preproc.
    - Saves TabPFN predictions, metrics, and classification report under out_dir/.
    Returns a dict with paths and objects.
    """
    out_dir = Path(out_dir) if out_dir is not None else default_tabpfn_out_dir(category, ct_name)
    preproc_dir = out_dir / "preproc"
    # Step 7
    X_train_p, X_val_p, scaler, pca = standardize_pca(
        X_train, X_val, n_components_max=n_components_max, random_state=random_state, save_dir=preproc_dir
    )
    # Step 8
    res = train_eval_tabpfn(
        X_train=X_train_p,
        y_train=y_train,
        X_val=X_val_p,
        y_val=y_val,
        ids_val=ids_val,
        out_dir=out_dir,
        device=device,
        tabpfn_src=tabpfn_src,
        clf_kwargs=clf_kwargs,
    )
    res.update({
        "out_dir": out_dir,
        "preproc_dir": preproc_dir,
        "scaler_path": preproc_dir / "scaler.joblib",
        "pca_path": preproc_dir / "pca.joblib",
        "X_train_p_path": preproc_dir / "X_train_p.npy",
        "X_val_p_path": preproc_dir / "X_val_p.npy",
    })
    return res
