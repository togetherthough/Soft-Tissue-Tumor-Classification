from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split

from med3pipe.sam.core import load_pooled_features


def _clean_id(cid: str) -> str:
    return cid[:-4] if cid.endswith('.nii') else cid


def stratified_features_split(
    feat_train_dir: Path,
    feat_val_dir: Path,
    labels_tr_dir: Path | None,
    labels_val_dir: Path | None,
    lab_map: Dict[str, int],
    train_ratio: float = 0.8,
    seed: int = 2025,
) -> Tuple[Tuple[np.ndarray, np.ndarray, List[str]], Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Build a stratified train/val split from the UNION of pooled features coming
    from the dataset's train and val feature directories.

    Returns: (X_train, y_train, ids_train), (X_val, y_val, ids_val)
    - X_* shape: (n_samples, n_features)
    - y_* dtype: int
    - ids_*: case_id strings aligned to rows
    """
    # Load features from both dirs
    X_parts: List[np.ndarray] = []
    id_parts: List[List[str]] = []

    Xtr, idtr = load_pooled_features(Path(feat_train_dir), labels_tr_dir)
    if Xtr.size:
        X_parts.append(Xtr)
        id_parts.append(idtr)

    Xva, idva = load_pooled_features(Path(feat_val_dir), labels_val_dir)
    if Xva.size:
        X_parts.append(Xva)
        id_parts.append(idva)

    if not X_parts:
        return (
            (np.empty((0,)), np.empty((0,), dtype=int), []),
            (np.empty((0,)), np.empty((0,), dtype=int), []),
        )

    X_all = X_parts[0] if len(X_parts) == 1 else np.vstack(X_parts)
    ids_all = sum(id_parts, [])
    ids_all_arr = np.array(ids_all)

    # Deduplicate by case ID (keep first occurrence)
    if len(ids_all_arr) > 1:
        _, first_idx = np.unique(ids_all_arr, return_index=True)
        first_idx = np.sort(first_idx)
        if len(first_idx) != len(ids_all_arr):
            X_all = X_all[first_idx]
            ids_all_arr = ids_all_arr[first_idx]

    # Filter to labeled subset only
    mask_lab = np.array([_clean_id(c) in lab_map for c in ids_all_arr], dtype=bool)
    if not mask_lab.any():
        return (
            (np.empty((0,)), np.empty((0,), dtype=int), []),
            (np.empty((0,)), np.empty((0,), dtype=int), []),
        )

    X_l = X_all[mask_lab]
    ids_l = ids_all_arr[mask_lab]
    y_l = np.array([lab_map[_clean_id(c)] for c in ids_l], dtype=int)

    # Compute split; prefer stratified if at least 2 classes and enough samples
    rs = np.random.RandomState(int(seed))
    unique_classes = np.unique(y_l)
    test_size = float(max(min(1 - float(train_ratio), 0.99), 0.01))

    # Require at least two classes for stratified sampling
    if unique_classes.size < 2:
        raise ValueError(
            "Stratified split requires at least two classes in the labeled set; found only one."
        )

    # Try sklearn stratified split first when feasible
    try:
        X_tr, X_va, y_tr, y_va, ids_tr, ids_va = train_test_split(
            X_l, y_l, ids_l, test_size=test_size, random_state=int(seed), stratify=y_l
        )
        # Ensure VAL has >= 2 classes
        if np.unique(y_va).size >= 2:
            return (X_tr, y_tr, ids_tr.tolist()), (X_va, y_va, ids_va.tolist())
    except Exception:
        pass

    # Build per-class index lists
    cls_to_idx = {c: np.where(y_l == c)[0].tolist() for c in unique_classes}
    for c in cls_to_idx:
        rs.shuffle(cls_to_idx[c])

    n_va_target = max(2, int(round(len(y_l) * test_size)))
    va_sel: List[int] = []

    # First, try to include at least one sample from as many classes as possible (up to target)
    classes_sorted = sorted(cls_to_idx.keys(), key=lambda c: -len(cls_to_idx[c]))
    for c in classes_sorted:
        if not cls_to_idx[c]:
            continue
        if len(va_sel) >= n_va_target:
            break
        va_sel.append(cls_to_idx[c].pop())

    # Ensure we have at least two distinct classes in VAL
    if len({int(y_l[i]) for i in va_sel}) < 2:
        # Add one more from a different class if available
        for c in classes_sorted:
            if not cls_to_idx[c]:
                continue
            # Pick if class differs from existing ones
            cand = cls_to_idx[c][-1]
            if int(y_l[cand]) not in {int(y_l[i]) for i in va_sel}:
                va_sel.append(cls_to_idx[c].pop())
                break

    # Fill remaining VAL slots randomly from remaining pool
    remaining = [i for c in classes_sorted for i in cls_to_idx[c]]
    rs.shuffle(remaining)
    while len(va_sel) < n_va_target and remaining:
        va_sel.append(remaining.pop())

    va_sel = sorted(set(va_sel))
    tr_sel = sorted(set(range(len(y_l))) - set(va_sel))

    X_tr, y_tr = X_l[tr_sel], y_l[tr_sel]
    X_va, y_va = X_l[va_sel], y_l[va_sel]
    ids_tr = ids_l[tr_sel]
    ids_va = ids_l[va_sel]

    return (X_tr, y_tr, ids_tr.tolist()), (X_va, y_va, ids_va.tolist())
