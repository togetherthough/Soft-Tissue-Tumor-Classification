from __future__ import annotations

"""
med3pipe.tabular.localpfn

LoCalPFN: Retrieval + (optional) fine-tuning on local neighborhoods for TabPFN.

What we implement here:
- k-NN retrieval with configurable k in the standardized + PCA space (Step 7 features).
- Per-query local context inference using TabPFN (equivalent to TabPFN-kNN when fine-tune is off).
- Optional lightweight adapter fine-tuning on top of TabPFN logits using retrieved neighborhoods.
  This provides practical local adaptation without requiring access to TabPFN's internal training API.

Notes:
- The original LoCalPFN paper performs end-to-end fine-tuning of the transformer using an approximate
  batching scheme. As the public TabPFN sklearn interface does not expose a stable training API,
  we provide an adapter-based fine-tuning that is dataset-specific and trained on local neighborhoods.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import json
import time
import datetime as _dt

import numpy as np
import joblib

from .tabpfn import ensure_tabpfn_on_sys_path, standardize_pca


# ------------------------------
# Utilities
# ------------------------------

def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def default_localpfn_out_dir(
    category: str = "gist",
    ct_name: str = "ct_GIST",
    base_dir: Optional[Path] = None,
) -> Path:
    base = (base_dir or Path.cwd()).resolve()
    out = base / "tabpfn_runs" / f"local_{category}_{ct_name}_{_timestamp()}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ------------------------------
# Config
# ------------------------------

@dataclass
class LocalPFNConfig:
    # Retrieval
    k: Optional[int] = None  # if None, uses min(10*sqrt(n_train), 1000)
    metric: str = "euclidean"
    # Inference
    device: Optional[str] = None  # "cuda" | "cpu" | None (auto)
    tabpfn_src: Optional[Path] = None
    clf_kwargs: Optional[Dict[str, Any]] = None
    # Adapter fine-tuning (on TabPFN logits)
    fit_adapter: bool = False
    adapter_epochs: int = 10
    adapter_lr: float = 5e-2
    adapter_weight_decay: float = 0.0
    adapter_num_queries: int = 1000  # number of training queries to build adapter set
    adapter_verbose: bool = True


# ------------------------------
# Retrieval
# ------------------------------

def _auto_k(n_train: int) -> int:
    return int(min(1000, max(1, 10 * np.sqrt(max(1, n_train)))))


def build_knn_index(X_train_p: np.ndarray, metric: str = "euclidean") -> "NearestNeighbors":
    # Local import to avoid heavy sklearn import at module load time
    try:
        from sklearn import set_config as _sk_set_config  # type: ignore
        try:
            _sk_set_config(skip_parameter_validation=False)  # type: ignore
        except TypeError:
            pass
    except Exception:
        pass
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(metric=metric)
    nn.fit(X_train_p)
    return nn


def retrieve_neighbors(
    knn: NearestNeighbors,
    X_query_p: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, distances) of shape (n_query, k)."""
    dists, idxs = knn.kneighbors(X_query_p, n_neighbors=k, return_distance=True)
    return idxs, dists


# ------------------------------
# Adapter: a lightweight logistic head on top of TabPFN logits
# ------------------------------

def _logit(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _train_adapter(
    logits: np.ndarray,  # shape (N, 1) or (N,) for binary
    y: np.ndarray,       # shape (N,)
    epochs: int = 10,
    lr: float = 5e-2,
    weight_decay: float = 0.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a 1D linear adapter z' = a*z + b minimizing BCE.

    Returns dict with learned params and training history.
    """
    z = logits.reshape(-1, 1).astype(np.float32)
    y = y.astype(np.float32)
    a = 1.0
    b = 0.0

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    history = []
    for ep in range(epochs):
        z_lin = a * z + b
        p = sigmoid(z_lin)
        # BCE loss
        eps = 1e-8
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        # gradients w.r.t a and b
        grad = p - y.reshape(-1, 1)
        grad_a = float(np.mean(grad * z)) + weight_decay * a
        grad_b = float(np.mean(grad))
        # update
        a -= lr * grad_a
        b -= lr * grad_b
        history.append(float(loss))
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            print(f"[adapter] epoch {ep+1}/{epochs} loss={loss:.4f} a={a:.3f} b={b:.3f}")

    return {"a": float(a), "b": float(b), "history": history}


def _apply_adapter(logits: np.ndarray, adapter: Optional[Dict[str, Any]]) -> np.ndarray:
    if not adapter:
        return logits
    a = adapter.get("a", 1.0)
    b = adapter.get("b", 0.0)
    return a * logits + b


# ------------------------------
# Core: Local inference and optional adapter fine-tuning
# ------------------------------

def localpfn_infer(
    X_train_p: np.ndarray,
    y_train: np.ndarray,
    X_val_p: np.ndarray,
    y_val: np.ndarray,
    ids_val: Sequence[str],
    out_dir: Optional[Path] = None,
    cfg: Optional[LocalPFNConfig] = None,
) -> Dict[str, Any]:
    """Perform LoCalPFN-style inference (k-NN retrieval + per-query TabPFN).

    If cfg.fit_adapter is True, also trains a lightweight adapter on training queries and applies it.
    Saves predictions/metrics similar to `train_eval_tabpfn`.
    """
    cfg = cfg or LocalPFNConfig()

    # Ensure TabPFN import path and classifier
    tabpfn_src = ensure_tabpfn_on_sys_path(cfg.tabpfn_src)
    # Work around environments where torch.version may be missing (older or minimal builds)
    try:
        import torch as _torch  # type: ignore
        if not hasattr(_torch, "version"):
            class _TorchVersion:
                cuda = None
            _torch.version = _TorchVersion()  # type: ignore[attr-defined]
    except Exception:
        pass
    from tabpfn.classifier import TabPFNClassifier  # type: ignore

    # Import sklearn metrics lazily to avoid environment issues during module import
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

    # Determine device
    device = cfg.device or ("cuda" if _has_cuda() else "cpu")
    clf_kwargs = cfg.clf_kwargs or {}

    # Build KNN index on X_train_p
    k = cfg.k or _auto_k(X_train_p.shape[0])
    knn = build_knn_index(X_train_p, metric=cfg.metric)

    # Prepare out_dir
    out_dir = Path(out_dir) if out_dir is not None else default_localpfn_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dump = {
        **asdict(cfg),
        "k_effective": int(k),
        "device": device,
        "tabpfn_src": str(tabpfn_src) if tabpfn_src else None,
        "shapes": {
            "X_train_p": list(map(int, X_train_p.shape)),
            "X_val_p": list(map(int, X_val_p.shape)),
        },
    }
    (out_dir / "localpfn_config.json").write_text(json.dumps(config_dump, indent=2))

    # Optional: adapter fitting on a subset of training queries
    adapter: Optional[Dict[str, Any]] = None
    if cfg.fit_adapter:
        print("[LoCalPFN] Training adapter on local neighborhoods...")
        # Sample training queries
        n_q = min(int(cfg.adapter_num_queries), X_train_p.shape[0])
        rng = np.random.default_rng(42)
        q_idx = rng.choice(X_train_p.shape[0], size=n_q, replace=False)
        X_q = X_train_p[q_idx]
        y_q = y_train[q_idx]
        # For each query, get neighbors (excluding itself if present)
        idxs, _ = retrieve_neighbors(knn, X_q, k=k)
        logits_list = []
        y_list = []
        clf = TabPFNClassifier(device=device, **clf_kwargs)
        for qi, neigh in zip(q_idx, idxs):
            ctx_idx = neigh
            # Avoid the query itself in the context if present
            ctx_idx = ctx_idx[ctx_idx != qi]
            if ctx_idx.size == 0:
                continue
            X_ctx = X_train_p[ctx_idx]
            y_ctx = y_train[ctx_idx]
            # Fit on neighborhood context, predict logits for the query
            clf.fit(X_ctx, y_ctx)
            # We need logits; get proba and convert to logits for binary
            proba = clf.predict_proba(X_train_p[qi:qi+1])
            if proba.shape[1] != 2:
                # Only binary supported for adapter training
                continue
            logit = _logit(proba[:, 1:2])  # shape (1,1)
            logits_list.append(logit[0, 0])
            # Ground-truth label for the query
            y_list.append(int(y_train[qi]))
        if len(logits_list) > 10:
            logits_arr = np.array(logits_list, dtype=np.float32)
            y_arr = np.array(y_list, dtype=np.float32)
            adapter = _train_adapter(
                logits=logits_arr,
                y=y_arr,
                epochs=cfg.adapter_epochs,
                lr=cfg.adapter_lr,
                weight_decay=cfg.adapter_weight_decay,
                verbose=cfg.adapter_verbose,
            )
            (out_dir / "localpfn_adapter.json").write_text(json.dumps(adapter, indent=2))
        else:
            print("[LoCalPFN] Skipping adapter training (insufficient samples).")

    # Inference on VAL using per-query local contexts
    print("[LoCalPFN] Running local-context inference on validation set...")
    t0 = time.time()
    idxs_val, _ = retrieve_neighbors(knn, X_val_p, k=k)
    # Instantiate classifier once
    clf = TabPFNClassifier(device=device, **clf_kwargs)

    y_pred = np.zeros((X_val_p.shape[0],), dtype=np.int64)
    proba_val: Optional[np.ndarray] = None
    proba_buf: list[np.ndarray] = []

    for i, neigh in enumerate(idxs_val):
        X_ctx = X_train_p[neigh]
        y_ctx = y_train[neigh]
        clf.fit(X_ctx, y_ctx)
        # predict_proba then optionally apply adapter on logits
        try:
            p = clf.predict_proba(X_val_p[i:i+1])  # shape (1,C)
            if adapter is not None and p.shape[1] == 2:
                logit = _logit(p[:, 1:2])  # (1,1)
                logit_adj = _apply_adapter(logit, adapter)
                # back to probability
                p1 = 1.0 / (1.0 + np.exp(-logit_adj))
                p = np.concatenate([1 - p1, p1], axis=1)
            proba_buf.append(p[0])
        except Exception:
            # Fall back to hard prediction
            pass
        y_pred[i] = int(clf.predict(X_val_p[i:i+1])[0])

    infer_time = time.time() - t0

    # Metrics
    acc = float(accuracy_score(y_val, y_pred))
    macro_f1 = float(f1_score(y_val, y_pred, average="macro"))
    roc_auc = None
    if proba_buf and len(proba_buf) == len(y_pred):
        proba_val = np.stack(proba_buf, axis=0)
        if proba_val.shape[1] == 2:
            roc_auc = float(roc_auc_score(y_val, proba_val[:, 1]))

    report_txt = classification_report(y_val, y_pred, target_names=["0", "1"])
    cm = confusion_matrix(y_val, y_pred)

    # Save predictions
    import pandas as pd
    pred_df: Dict[str, Any] = {
        "case_id": list(ids_val),
        "true_label": list(map(int, y_val)),
        "pred_label": list(map(int, y_pred)),
    }
    if proba_val is not None:
        pred_df["proba_0"] = proba_val[:, 0].tolist()
        pred_df["proba_1"] = proba_val[:, 1].tolist()
    pd.DataFrame(pred_df).to_csv(out_dir / "localpfn_val_predictions.csv", index=False)

    # Save metrics
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.astype(int).tolist(),
        "infer_time_sec": infer_time,
        "adapter": adapter,
    }
    (out_dir / "localpfn_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "localpfn_classification_report.txt").write_text(report_txt)

    return {
        "metrics": metrics,
        "pred_path": out_dir / "localpfn_val_predictions.csv",
        "metrics_path": out_dir / "localpfn_metrics.json",
        "config_path": out_dir / "localpfn_config.json",
        "report_path": out_dir / "localpfn_classification_report.txt",
        "out_dir": out_dir,
    }


def _has_cuda() -> bool:
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


# ------------------------------
# High-level pipeline (Step 7 + LoCalPFN on top)
# ------------------------------

def localpfn_pipeline_single_fold(
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
    cfg: Optional[LocalPFNConfig] = None,
) -> Dict[str, Any]:
    """Run Step 7 (Standardize+PCA) then LoCalPFN inference for a single fold.

    Artifacts are saved under out_dir (created if missing). Returns a dict with paths and metrics.
    """
    out_dir = Path(out_dir) if out_dir is not None else default_localpfn_out_dir(category, ct_name)
    preproc_dir = out_dir / "preproc"

    # Step 7: Standardize + PCA (fit on TRAIN)
    X_train_p, X_val_p, scaler, pca = standardize_pca(
        X_train,
        X_val,
        n_components_max=n_components_max,
        random_state=random_state,
        save_dir=preproc_dir,
    )

    # LoCalPFN inference (+ optional adapter training)
    res = localpfn_infer(
        X_train_p=X_train_p,
        y_train=y_train,
        X_val_p=X_val_p,
        y_val=y_val,
        ids_val=ids_val,
        out_dir=out_dir,
        cfg=cfg or LocalPFNConfig(),
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


def localpfn_pipeline(
    folds: list,
    category: str = "gist",
    ct_name: str = "ct_GIST",
    out_dir: Optional[Path] = None,
    n_components_max: int = 500,
    random_state: int = 42,
    cfg: Optional[LocalPFNConfig] = None,
) -> Dict[str, Any]:
    """Run k-fold LoCalPFN pipeline and aggregate results.

    Args:
        folds: List of ((X_train, y_train, ids_train), (X_val, y_val, ids_val)) tuples from k-fold split
        category, ct_name: Dataset identifiers
        out_dir: Base output directory (fold-specific subdirs will be created)
        ... other LoCalPFN parameters

    Returns:
        Dict with aggregated metrics and paths to all fold results
    """
    if not folds:
        raise ValueError("No folds provided to localpfn_pipeline")
    
    base_out_dir = Path(out_dir) if out_dir is not None else default_localpfn_out_dir(category, ct_name)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    fold_results = []
    all_accuracies = []
    all_f1_scores = []
    all_roc_aucs = []
    
    print(f"\n{'='*60}")
    print(f"Running {len(folds)}-fold cross-validation for LoCalPFN")
    print(f"{'='*60}\n")
    
    for fold_idx, ((X_train, y_train, ids_train), (X_val, y_val, ids_val)) in enumerate(folds, 1):
        print(f"\n--- Fold {fold_idx}/{len(folds)} ---")
        
        fold_out_dir = base_out_dir / f"fold_{fold_idx}"
        fold_res = localpfn_pipeline_single_fold(
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
            cfg=cfg,
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
