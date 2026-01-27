from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple, List, Optional, Set

import numpy as np
from sklearn.model_selection import StratifiedKFold

from med3pipe.sam.core import load_pooled_features
from .lesion_filter import LesionSizeFilter


def _clean_id(cid: str) -> str:
    return cid[:-4] if cid.endswith('.nii') else cid


def stratified_kfold_features(
    feat_train_dir: Path,
    feat_val_dir: Path,
    labels_tr_dir: Path | None,
    labels_val_dir: Path | None,
    lab_map: Dict[str, int],
    n_splits: int = 5,
    seed: int = 2025,
    lesion_filter: Optional[LesionSizeFilter] = None,
    pooling_strategy: str = 'avg',
) -> List[Tuple[Tuple[np.ndarray, np.ndarray, List[str]], Tuple[np.ndarray, np.ndarray, List[str]]]]:
    """
    Build k-fold stratified splits from the UNION of pooled features coming
    from the dataset's train and val feature directories.

    Args:
        pooling_strategy: Pooling strategy to use ('avg', 'multiscale', or 'percentile')

    Returns: List of (train_data, val_data) tuples for each fold, where:
    - train_data = (X_train, y_train, ids_train)
    - val_data = (X_val, y_val, ids_val)
    - X_* shape: (n_samples, n_features)
    - y_* dtype: int
    - ids_*: case_id strings aligned to rows
    """
    # Load features from both dirs
    X_parts: List[np.ndarray] = []
    id_parts: List[List[str]] = []

    Xtr, idtr = load_pooled_features(Path(feat_train_dir), labels_tr_dir, pooling_strategy=pooling_strategy)
    if Xtr.size:
        X_parts.append(Xtr)
        id_parts.append(idtr)

    Xva, idva = load_pooled_features(Path(feat_val_dir), labels_val_dir, pooling_strategy=pooling_strategy)
    if Xva.size:
        X_parts.append(Xva)
        id_parts.append(idva)

    if not X_parts:
        return []

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
        return []

    X_l = X_all[mask_lab]
    ids_l = ids_all_arr[mask_lab]
    y_l = np.array([lab_map[_clean_id(c)] for c in ids_l], dtype=int)
    
    # Apply lesion size filtering if enabled
    if lesion_filter is not None and lesion_filter.is_enabled():
        lesion_filter.print_filter_summary()
        valid_cases = lesion_filter.get_valid_cases()
        
        # Filter to valid lesion sizes
        mask_lesion = np.array([_clean_id(c) in valid_cases for c in ids_l], dtype=bool)
        filtered_count = (~mask_lesion).sum()
        
        if filtered_count > 0:
            print(f"[INFO] Filtered out {filtered_count} cases based on lesion size criteria")
        
        if not mask_lesion.any():
            print("[WARNING] No cases remaining after lesion size filtering!")
            return []
        
        X_l = X_l[mask_lesion]
        ids_l = ids_l[mask_lesion]
        y_l = y_l[mask_lesion]

    # Require at least two classes for stratified sampling
    unique_classes = np.unique(y_l)
    if unique_classes.size < 2:
        raise ValueError(
            "Stratified k-fold requires at least two classes in the labeled set; found only one."
        )

    # Create StratifiedKFold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    
    folds = []
    for train_idx, val_idx in skf.split(X_l, y_l):
        X_tr, y_tr = X_l[train_idx], y_l[train_idx]
        X_va, y_va = X_l[val_idx], y_l[val_idx]
        ids_tr = ids_l[train_idx]
        ids_va = ids_l[val_idx]
        
        folds.append(
            ((X_tr, y_tr, ids_tr.tolist()), (X_va, y_va, ids_va.tolist()))
        )
    
    return folds


def stratified_features_split(
    feat_train_dir: Path,
    feat_val_dir: Path,
    labels_tr_dir: Path | None,
    labels_val_dir: Path | None,
    lab_map: Dict[str, int],
    train_ratio: float = 0.8,
    seed: int = 2025,
    lesion_filter: Optional[LesionSizeFilter] = None,
    n_splits: int = 5,
    pooling_strategy: str = 'avg',
) -> List[Tuple[Tuple[np.ndarray, np.ndarray, List[str]], Tuple[np.ndarray, np.ndarray, List[str]]]]:
    """
    Build k-fold stratified splits from the UNION of pooled features coming
    from the dataset's train and val feature directories.
    
    Args:
        pooling_strategy: Pooling strategy to use ('avg', 'multiscale', or 'percentile')
    
    NOTE: train_ratio parameter is deprecated and ignored. K-fold split is used instead.

    Returns: List of (train_data, val_data) tuples for each fold
    - train_data = (X_train, y_train, ids_train)
    - val_data = (X_val, y_val, ids_val)
    - X_* shape: (n_samples, n_features)
    - y_* dtype: int
    - ids_*: case_id strings aligned to rows
    """
    return stratified_kfold_features(
        feat_train_dir=feat_train_dir,
        feat_val_dir=feat_val_dir,
        labels_tr_dir=labels_tr_dir,
        labels_val_dir=labels_val_dir,
        lab_map=lab_map,
        n_splits=n_splits,
        seed=seed,
        lesion_filter=lesion_filter,
        pooling_strategy=pooling_strategy,
    )
        seed=seed,
        lesion_filter=lesion_filter,
    )
