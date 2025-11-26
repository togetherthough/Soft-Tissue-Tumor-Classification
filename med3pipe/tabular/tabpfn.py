from __future__ import annotations

"""
med3pipe.tabular.tabpfn

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
    
    # Print dimensions before PCA
    print(f"\n{'='*60}")
    print(f"ENCODER OUTPUT DIMENSIONS (before PCA):")
    print(f"  Train shape: {X_train.shape} (samples x features)")
    print(f"  Val shape:   {X_val.shape} (samples x features)")
    print(f"{'='*60}\n")
    # Import sklearn components here to avoid heavy import at module level
    try:
        from sklearn import set_config as _sk_set_config  # type: ignore
        try:
            _sk_set_config(skip_parameter_validation=False)  # type: ignore
        except TypeError:
            # Older scikit-learn versions: argument not supported
            pass
    except Exception:
        # set_config not available; continue
        pass
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

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

    # Work around environments where torch.version may be missing (older or minimal builds)
    try:
        import torch as _torch  # type: ignore
        if not hasattr(_torch, "version"):
            class _TorchVersion:  # minimal shim for torch.version
                cuda = None
            _torch.version = _TorchVersion()  # type: ignore[attr-defined]
    except Exception:
        pass

    # Lazy import TabPFN
    from tabpfn.classifier import TabPFNClassifier  # type: ignore

    # Lazy import sklearn metrics to avoid heavy import at module import time
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            classification_report,
            confusion_matrix,
            roc_auc_score,
        )
    except Exception:
        try:
            from sklearn import set_config as _sk_set_config  # type: ignore
            try:
                _sk_set_config(skip_parameter_validation=False)  # type: ignore
            except TypeError:
                pass
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                classification_report,
                confusion_matrix,
                roc_auc_score,
            )
        except Exception as e:
            raise ImportError(
                f"Failed to import sklearn.metrics due to environment configuration: {e}. "
                "Please ensure scikit-learn is correctly installed in this environment."
            )

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
        if proba is not None:
            if proba.ndim == 1:
                # Ensure 2D [n_samples, n_classes]
                proba = np.stack([1 - proba, proba], axis=-1)
            if proba.shape[1] == 2:
                # Binary ROC AUC
                roc_auc = float(roc_auc_score(y_val, proba[:, 1]))
            elif proba.shape[1] > 2:
                # Multiclass ROC AUC (macro, one-vs-rest)
                try:
                    # Determine classifier class order and restrict to classes present in y_val
                    if hasattr(clf, "classes_") and len(clf.classes_) == proba.shape[1]:
                        cls_arr = np.array(clf.classes_)
                    else:
                        # Fallback: assume classes are 0..K-1 in column order
                        cls_arr = np.arange(proba.shape[1])
                    present = np.unique(y_val)
                    mask = np.isin(cls_arr, present)
                    # Need at least 2 classes present to compute multiclass AUC
                    if mask.sum() >= 2:
                        proba_present = proba[:, mask]
                        labels_present = cls_arr[mask]
                        try:
                            roc_auc = float(roc_auc_score(y_val, proba_present, multi_class="ovr", labels=labels_present))
                        except Exception:
                            # Manual fallback: mean of one-vs-rest AUCs for classes present
                            aucs = []
                            for j, cls in enumerate(labels_present):
                                y_bin = (y_val == cls).astype(int)
                                # Need both positive and negative samples
                                if y_bin.min() == y_bin.max():
                                    continue
                                try:
                                    aucs.append(roc_auc_score(y_bin, proba_present[:, j]))
                                except Exception:
                                    continue
                            roc_auc = float(np.mean(aucs)) if aucs else None
                    else:
                        roc_auc = None
                except Exception:
                    roc_auc = None
    except Exception:
        proba = None
    
    # Derive label set dynamically to support binary or multi-class
    labels_sorted = np.unique(np.concatenate([np.unique(y_val), np.unique(y_pred)]))
    report_txt = classification_report(y_val, y_pred, labels=labels_sorted)
    cm = confusion_matrix(y_val, y_pred, labels=labels_sorted)

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
        "classes": (clf.classes_.tolist() if hasattr(clf, "classes_") else None),
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
        # Save all class probability columns with consistent naming
        for k in range(proba.shape[1]):
            pred_df[f"proba_k{k}"] = proba[:, k].tolist()
        # Also save label-named columns if classes_ available
        try:
            if hasattr(clf, "classes_") and len(clf.classes_) == proba.shape[1]:
                for idx, cls in enumerate(clf.classes_):
                    pred_df[f"proba_c{cls}"] = proba[:, idx].tolist()
        except Exception:
            pass
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

def tabpfn_pipeline_single_fold(
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
    """Run Step 7 (Standardize+PCA) and Step 8 (TabPFN train/eval) for a single fold.

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


def tabpfn_pipeline(
    folds: list,
    category: str = "gist",
    ct_name: str = "ct_GIST",
    out_dir: Optional[Path] = None,
    n_components_max: int = 500,
    random_state: int = 42,
    device: Optional[str] = None,
    tabpfn_src: Optional[Path] = None,
    clf_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run k-fold TabPFN pipeline and aggregate results.

    Args:
        folds: List of ((X_train, y_train, ids_train), (X_val, y_val, ids_val)) tuples from k-fold split
        category, ct_name: Dataset identifiers
        out_dir: Base output directory (fold-specific subdirs will be created)
        ... other TabPFN parameters

    Returns:
        Dict with aggregated metrics and paths to all fold results
    """
    if not folds:
        raise ValueError("No folds provided to tabpfn_pipeline")
    
    base_out_dir = Path(out_dir) if out_dir is not None else default_tabpfn_out_dir(category, ct_name)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    fold_results = []
    all_accuracies = []
    all_f1_scores = []
    all_roc_aucs = []
    
    print(f"\n{'='*60}")
    print(f"Running {len(folds)}-fold cross-validation for TabPFN")
    print(f"{'='*60}\n")
    
    for fold_idx, ((X_train, y_train, ids_train), (X_val, y_val, ids_val)) in enumerate(folds, 1):
        print(f"\n--- Fold {fold_idx}/{len(folds)} ---")
        
        fold_out_dir = base_out_dir / f"fold_{fold_idx}"
        fold_res = tabpfn_pipeline_single_fold(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            ids_val=ids_val,
            category=category,
            ct_name=ct_name,
            out_dir=fold_out_dir,
            n_components_max=n_components_max,
            random_state=random_state,
            device=device,
            tabpfn_src=tabpfn_src,
            clf_kwargs=clf_kwargs,
        )
        
        fold_results.append(fold_res)
        all_accuracies.append(fold_res['metrics']['accuracy'])
        all_f1_scores.append(fold_res['metrics']['macro_f1'])
        if fold_res['metrics'].get('roc_auc') is not None:
            all_roc_aucs.append(fold_res['metrics']['roc_auc'])
        
        print(f"Fold {fold_idx} - Accuracy: {fold_res['metrics']['accuracy']:.4f}, "
              f"F1: {fold_res['metrics']['macro_f1']:.4f}")
    
    # Aggregate metrics
    aggregated_metrics = {
        'accuracy': float(np.mean(all_accuracies)),
        'accuracy_std': float(np.std(all_accuracies)),
        'macro_f1': float(np.mean(all_f1_scores)),
        'macro_f1_std': float(np.std(all_f1_scores)),
        'roc_auc': float(np.mean(all_roc_aucs)) if all_roc_aucs else None,
        'roc_auc_std': float(np.std(all_roc_aucs)) if all_roc_aucs else None,
        'n_folds': len(folds),
        'fold_accuracies': all_accuracies,
        'fold_f1_scores': all_f1_scores,
        'fold_roc_aucs': all_roc_aucs if all_roc_aucs else None,
    }
    
    # Save aggregated metrics
    with open(base_out_dir / "kfold_summary.json", "w") as f:
        json.dump(aggregated_metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"K-Fold Cross-Validation Summary")
    print(f"{'='*60}")
    print(f"Accuracy:  {aggregated_metrics['accuracy']:.4f} ± {aggregated_metrics['accuracy_std']:.4f}")
    print(f"F1 Score:  {aggregated_metrics['macro_f1']:.4f} ± {aggregated_metrics['macro_f1_std']:.4f}")
    if aggregated_metrics['roc_auc'] is not None:
        print(f"ROC AUC:   {aggregated_metrics['roc_auc']:.4f} ± {aggregated_metrics['roc_auc_std']:.4f}")
    print(f"{'='*60}\n")
    
    return {
        'metrics': aggregated_metrics,
        'fold_results': fold_results,
        'out_dir': base_out_dir,
        'summary_path': base_out_dir / "kfold_summary.json",
    }
