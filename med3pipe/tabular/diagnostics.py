"""
med3pipe.tabular.diagnostics

Statistical diagnostic tests for evaluating embedding quality before classification.

Three complementary tests at increasing granularity:
1. Mann-Whitney U  (feature-level)  — are individual dimensions diagnostic?
2. HSIC / CKA      (global)         — does the representation contain label info at all?
3. kNN agreement    (local geometry) — can similar patients help classify new ones?

All functions are self-contained and operate on (X, y) arrays directly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Test 1: Mann-Whitney U (per-dimension)
# ---------------------------------------------------------------------------

def mann_whitney_test(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run Mann-Whitney U test on each feature dimension independently.

    For each of the D dimensions, test H₀: the distribution of values is the
    same for class 0 and class 1.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Binary label vector, shape (n_samples,) with values in {0, 1}.
        alpha: Significance level for FDR correction.

    Returns:
        DataFrame with columns:
            dimension   – feature index (0-based)
            u_stat      – Mann-Whitney U statistic
            p_value     – two-sided p-value
            p_value_fdr – Benjamini-Hochberg corrected p-value
            significant – bool, True if p_value_fdr < alpha
            effect_size – rank-biserial correlation r = 1 − 2U/(n₀·n₁)
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if len(y) != X.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples but y has {len(y)}")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError(f"y must contain only 0 and 1, got unique values {np.unique(y)}")

    mask0 = y == 0
    mask1 = y == 1
    n0, n1 = mask0.sum(), mask1.sum()
    if n0 == 0 or n1 == 0:
        raise ValueError("Both classes must have at least one sample")

    n_features = X.shape[1]
    u_stats = np.empty(n_features)
    p_values = np.empty(n_features)

    for d in range(n_features):
        u, p = stats.mannwhitneyu(X[mask0, d], X[mask1, d], alternative="two-sided")
        u_stats[d] = u
        p_values[d] = p

    # Benjamini-Hochberg FDR correction
    p_values_fdr = _benjamini_hochberg(p_values)

    # Effect size: rank-biserial correlation
    effect_sizes = 1.0 - (2.0 * u_stats) / (n0 * n1)

    return pd.DataFrame({
        "dimension": np.arange(n_features),
        "u_stat": u_stats,
        "p_value": p_values,
        "p_value_fdr": p_values_fdr,
        "significant": p_values_fdr < alpha,
        "effect_size": effect_sizes,
    })


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    p = np.asarray(p_values)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)

    adjusted = p * n / ranks
    # Enforce monotonicity: walk backwards through the sorted order
    adjusted_sorted = adjusted[order]
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i + 1] = min(adjusted_sorted[i + 1], 1.0)
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted_sorted[-1] = min(adjusted_sorted[-1], 1.0)

    result = np.empty(n)
    result[order] = adjusted_sorted
    return result


# ---------------------------------------------------------------------------
# Test 2: HSIC / CKA (global dependence)
# ---------------------------------------------------------------------------

def hsic_test(
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 10_000,
    kernel_bandwidth: Optional[float] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """Hilbert-Schmidt Independence Criterion with permutation p-value.

    Measures statistical dependence between feature matrix X and label vector y
    using kernel methods. Captures any form of dependence (linear, nonlinear).

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Binary label vector, shape (n_samples,).
        n_permutations: Number of label shuffles for the permutation test.
        kernel_bandwidth: RBF bandwidth for X kernel. If None, uses median heuristic.
        random_state: Seed for reproducibility.

    Returns:
        Dict with keys:
            hsic       – observed HSIC statistic
            cka        – Centered Kernel Alignment (normalized HSIC)
            p_value    – permutation p-value
            n_perms    – number of permutations used
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    n = len(y)
    if X.shape[0] != n:
        raise ValueError(f"X has {X.shape[0]} samples but y has {n}")

    # Kernel for X: RBF with median heuristic
    K_X = _rbf_kernel(X, bandwidth=kernel_bandwidth)

    # Kernel for y: delta kernel (1 if same label, 0 otherwise)
    K_Y = (y[:, None] == y[None, :]).astype(np.float64)

    # Center kernels
    H = np.eye(n) - np.ones((n, n)) / n
    K_X_c = H @ K_X @ H
    K_Y_c = H @ K_Y @ H

    # HSIC = (1/(n-1)^2) * trace(K_X_c @ K_Y_c)
    observed_hsic = np.trace(K_X_c @ K_Y_c) / ((n - 1) ** 2)

    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    hsic_xx = np.trace(K_X_c @ K_X_c) / ((n - 1) ** 2)
    hsic_yy = np.trace(K_Y_c @ K_Y_c) / ((n - 1) ** 2)
    denom = np.sqrt(hsic_xx * hsic_yy)
    cka = observed_hsic / denom if denom > 0 else 0.0

    # Permutation test: shuffle y, recompute HSIC
    rng = np.random.RandomState(random_state)
    null_hsics = np.empty(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        K_Y_perm = (y_perm[:, None] == y_perm[None, :]).astype(np.float64)
        K_Y_perm_c = H @ K_Y_perm @ H
        null_hsics[i] = np.trace(K_X_c @ K_Y_perm_c) / ((n - 1) ** 2)

    p_value = (np.sum(null_hsics >= observed_hsic) + 1) / (n_permutations + 1)

    return {
        "hsic": float(observed_hsic),
        "cka": float(cka),
        "p_value": float(p_value),
        "n_perms": n_permutations,
    }


def _rbf_kernel(X: np.ndarray, bandwidth: Optional[float] = None) -> np.ndarray:
    """Compute RBF (Gaussian) kernel matrix with optional median heuristic."""
    from sklearn.metrics.pairwise import euclidean_distances

    dists = euclidean_distances(X, X)
    if bandwidth is None:
        # Median heuristic: bandwidth = median of nonzero pairwise distances
        nonzero = dists[np.triu_indices_from(dists, k=1)]
        bandwidth = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0
        bandwidth = max(bandwidth, 1e-10)  # avoid division by zero
    return np.exp(-dists ** 2 / (2 * bandwidth ** 2))


# ---------------------------------------------------------------------------
# Test 3: kNN Label Agreement (local geometry)
# ---------------------------------------------------------------------------

def knn_agreement_test(
    X: np.ndarray,
    y: np.ndarray,
    k_values: Sequence[int] = (3, 5, 7, 10, 15, 20),
    n_permutations: int = 1_000,
    standardize: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Test whether nearest neighbors in feature space share labels.

    For each sample, finds its k nearest neighbors and computes the fraction
    that share the same label. Compares to a permutation null distribution.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Binary label vector, shape (n_samples,).
        k_values: Sequence of k values to test.
        n_permutations: Number of label shuffles for p-value estimation.
        standardize: If True, z-score standardize features before computing distances.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with columns:
            k               – number of neighbors
            mean_agreement  – observed mean label agreement
            expected_chance – expected agreement under random labels (majority class prior)
            adjusted_agreement – (observed - expected) / (1 - expected), like Cohen's kappa
            p_value         – permutation p-value
            per_sample_agreement – list of per-sample agreement values (for further analysis)
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    n = len(y)
    if X.shape[0] != n:
        raise ValueError(f"X has {X.shape[0]} samples but y has {n}")

    # Standardize if requested
    if standardize:
        mu = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0  # avoid div by zero for constant features
        X = (X - mu) / std

    # Expected chance agreement = sum of squared class proportions
    # (probability that two random draws match)
    class_props = np.bincount(y, minlength=2) / n
    expected_chance = float(np.sum(class_props ** 2))

    max_k = max(k for k in k_values if k < n)
    if max_k <= 0:
        return pd.DataFrame(columns=["k", "mean_agreement", "expected_chance",
                                      "adjusted_agreement", "p_value",
                                      "per_sample_agreement"])
    # k+1 because the query point itself is included in the results
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    # Remove self (first column is always self with distance 0)
    neighbor_indices = indices[:, 1:]

    rng = np.random.RandomState(random_state)

    rows = []
    for k in k_values:
        if k > n - 1:
            continue  # skip if k >= number of samples
        knn_idx = neighbor_indices[:, :k]  # (n, k)
        knn_labels = y[knn_idx]  # (n, k)

        # Per-sample agreement: fraction of k neighbors with same label
        per_sample = np.mean(knn_labels == y[:, None], axis=1)  # (n,)
        observed = float(np.mean(per_sample))

        # Permutation test
        null_agreements = np.empty(n_permutations)
        for i in range(n_permutations):
            y_perm = rng.permutation(y)
            perm_labels = y_perm[knn_idx]
            perm_agreement = np.mean(perm_labels == y_perm[:, None])
            null_agreements[i] = perm_agreement

        p_value = (np.sum(null_agreements >= observed) + 1) / (n_permutations + 1)

        # Adjusted agreement (Cohen's kappa analog)
        if expected_chance < 1.0:
            adjusted = (observed - expected_chance) / (1.0 - expected_chance)
        else:
            adjusted = 0.0

        rows.append({
            "k": k,
            "mean_agreement": observed,
            "expected_chance": expected_chance,
            "adjusted_agreement": adjusted,
            "p_value": p_value,
            "per_sample_agreement": per_sample.tolist(),
        })

    return pd.DataFrame(rows)
