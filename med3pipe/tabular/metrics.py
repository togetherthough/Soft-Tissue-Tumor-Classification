from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray, pos_label: int) -> float | None:
    y_bin = (y_true == pos_label).astype(int)
    # Need both positives and negatives
    if y_bin.min() == y_bin.max():
        return None
    try:
        return float(roc_auc_score(y_bin, y_score))
    except Exception:
        return None


def per_dataset_breakdown(
    pred_csv_path: Path | str,
    ids_val: Sequence[str],
    dset_val: Sequence[str],
    config_path: Path | str,
) -> pd.DataFrame:
    """
    Build a per-dataset metrics table (accuracy, macro_f1, roc_auc, n) that is
    robust to binary vs multiclass and aligns probabilities to the correct
    classes using the classifier's recorded class order.

    - pred_csv_path: path to tabpfn_val_predictions.csv
    - ids_val: sequence of validation case_ids (same order as passed to training)
    - dset_val: sequence of dataset names aligned to ids_val
    - config_path: path to tabpfn_config.json containing `classes`
    """
    pred_csv_path = Path(pred_csv_path)
    config_path = Path(config_path)

    pred = pd.read_csv(pred_csv_path)
    # Attach dataset names to predictions using the ids order passed to training
    map_df = pd.DataFrame({"case_id": list(ids_val), "dataset": list(dset_val)})
    pred = pred.merge(map_df, on="case_id", how="left")

    cfg = json.loads(Path(config_path).read_text())
    classes = np.array(cfg.get("classes") or [])  # could be empty

    # Determine probability column scheme
    has_label_cols = any(c.startswith("proba_c") for c in pred.columns)
    has_k_cols = any(c.startswith("proba_k") for c in pred.columns)

    rows: list[dict] = []
    for dname, g in pred.groupby("dataset"):
        y_true = g["true_label"].to_numpy()
        y_pred = g["pred_label"].to_numpy()
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro"))

        auc: float | None = None
        present = np.unique(y_true)

        try:
            if present.size >= 2:
                # Binary case: compute AUC with a single positive class column
                if present.size == 2:
                    # pick positive label: prefer 1 if present, else max label
                    pos = 1 if 1 in present else int(np.max(present))
                    # Shortcut: support explicit proba_0/proba_1
                    if pos == 1 and "proba_1" in g.columns:
                        auc = _binary_auc(y_true, g["proba_1"].to_numpy(), pos)
                    elif pos == 0 and "proba_0" in g.columns:
                        auc = _binary_auc(y_true, g["proba_0"].to_numpy(), pos)
                    # Try label-named columns first
                    if auc is None and has_label_cols and classes.size and pos in classes:
                        pos_col = f"proba_c{int(pos)}"
                        if pos_col in g.columns:
                            auc = _binary_auc(y_true, g[pos_col].to_numpy(), pos)
                    # Fallback: k-indexed columns using class order
                    if auc is None and has_k_cols:
                        if classes.size:
                            # Map pos to class index
                            try:
                                idx = int(np.where(classes == pos)[0][0])
                                pos_col = f"proba_k{idx}"
                                if pos_col in g.columns:
                                    auc = _binary_auc(y_true, g[pos_col].to_numpy(), pos)
                            except Exception:
                                auc = None
                        else:
                            # Assume class order 0..K-1 corresponds to sorted unique labels
                            K = sum(c.startswith("proba_k") for c in pred.columns)
                            cls_arr = np.arange(K)
                            # If pos within range, use that column
                            if 0 <= pos < K:
                                pos_col = f"proba_k{pos}"
                                if pos_col in g.columns:
                                    auc = _binary_auc(y_true, g[pos_col].to_numpy(), pos)
                else:
                    # Multiclass AUC (macro OVR) over present classes only
                    if has_label_cols and classes.size and np.all(np.isin(present, classes)):
                        mask = np.isin(classes, present)
                        if mask.sum() >= 2:
                            cols = [f"proba_c{int(cls)}" for cls in classes[mask]]
                            y_score = g[cols].to_numpy()
                            auc = float(roc_auc_score(y_true, y_score, multi_class="ovr", labels=classes[mask]))
                    elif has_k_cols:
                        K = sum(c.startswith("proba_k") for c in pred.columns)
                        cls_arr = np.arange(K)
                        mask = np.isin(cls_arr, present)
                        if mask.sum() >= 2:
                            cols = [f"proba_k{k}" for k in np.where(mask)[0]]
                            y_score = g[cols].to_numpy()
                            auc = float(roc_auc_score(y_true, y_score, multi_class="ovr", labels=cls_arr[mask]))
        except Exception:
            auc = None

        rows.append({
            "dataset": dname,
            "accuracy": acc,
            "macro_f1": f1,
            "roc_auc": auc,
            "n": int(len(g)),
        })

    df = pd.DataFrame(rows).sort_values("dataset")
    return df
