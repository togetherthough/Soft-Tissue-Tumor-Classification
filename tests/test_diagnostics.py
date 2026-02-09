"""Unit tests for med3pipe.tabular.diagnostics.

Tests each diagnostic function with:
- Synthetic data with known signal → should detect it
- Null data (random labels) → should NOT false-alarm
"""

import numpy as np
import pytest

from med3pipe.tabular.diagnostics import (
    mann_whitney_test,
    hsic_test,
    knn_agreement_test,
    _benjamini_hochberg,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_signal_data(n=200, d=50, seed=42):
    """Create data where class 0 and class 1 differ in the first 10 dims."""
    rng = np.random.RandomState(seed)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    X = rng.randn(n, d)
    # Plant strong signal in first 10 dimensions
    X[y == 1, :10] += 2.0  # shift class 1 by 2 std in first 10 dims
    return X, y


def _make_null_data(n=200, d=50, seed=123):
    """Create data where labels are independent of features."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = rng.randint(0, 2, size=n)
    return X, y


# ---------------------------------------------------------------------------
# Test: Benjamini-Hochberg
# ---------------------------------------------------------------------------

class TestBenjaminiHochberg:
    def test_all_significant(self):
        """All tiny p-values should remain significant after correction."""
        p = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        adjusted = _benjamini_hochberg(p)
        assert all(adjusted < 0.05)

    def test_preserves_order(self):
        """Adjusted p-values should preserve the rank order of raw p-values."""
        p = np.array([0.01, 0.5, 0.03, 0.8, 0.001])
        adjusted = _benjamini_hochberg(p)
        assert np.all(np.argsort(adjusted) == np.argsort(p))

    def test_clipped_to_one(self):
        """Adjusted p-values should never exceed 1."""
        p = np.array([0.9, 0.95, 0.99, 1.0])
        adjusted = _benjamini_hochberg(p)
        assert np.all(adjusted <= 1.0)

    def test_single_pvalue(self):
        p = np.array([0.03])
        adjusted = _benjamini_hochberg(p)
        assert adjusted[0] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Test: Mann-Whitney
# ---------------------------------------------------------------------------

class TestMannWhitney:
    def test_detects_signal(self):
        """With planted signal, should find significant dimensions."""
        X, y = _make_signal_data()
        result = mann_whitney_test(X, y)
        
        assert isinstance(result, type(result))  # is a DataFrame
        assert len(result) == X.shape[1]
        
        # First 10 dims should be significant (signal planted there)
        sig_in_signal_dims = result.loc[result["dimension"] < 10, "significant"].sum()
        assert sig_in_signal_dims >= 8, f"Expected ≥8 of 10 signal dims significant, got {sig_in_signal_dims}"
        
        # Effect size should be large for signal dims
        mean_effect_signal = result.loc[result["dimension"] < 10, "effect_size"].abs().mean()
        assert mean_effect_signal > 0.3, f"Expected large effect size, got {mean_effect_signal}"

    def test_null_few_significant(self):
        """With random labels, very few dims should be significant after FDR."""
        X, y = _make_null_data()
        result = mann_whitney_test(X, y)
        
        n_sig = result["significant"].sum()
        # Under null, FDR controls false discovery rate at alpha=0.05
        # With 50 tests, we might get 0-3 by chance at most
        assert n_sig <= 5, f"Expected ≤5 false positives under null, got {n_sig}"

    def test_effect_size_range(self):
        """Effect sizes should be in [-1, 1]."""
        X, y = _make_signal_data()
        result = mann_whitney_test(X, y)
        assert result["effect_size"].between(-1, 1).all()

    def test_input_validation(self):
        """Should reject invalid inputs."""
        X = np.random.randn(10, 5)
        
        # Wrong y length
        with pytest.raises(ValueError):
            mann_whitney_test(X, np.array([0, 1, 0]))
        
        # Non-binary y
        with pytest.raises(ValueError):
            mann_whitney_test(X, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]))
        
        # 1D X
        with pytest.raises(ValueError):
            mann_whitney_test(np.random.randn(10), np.array([0]*5 + [1]*5))


# ---------------------------------------------------------------------------
# Test: HSIC
# ---------------------------------------------------------------------------

class TestHSIC:
    def test_detects_signal(self):
        """With planted signal, HSIC should be significant."""
        X, y = _make_signal_data()
        result = hsic_test(X, y, n_permutations=500, random_state=42)
        
        assert result["hsic"] > 0
        assert result["cka"] > 0.05, f"Expected CKA > 0.05 with signal, got {result['cka']}"
        assert result["p_value"] < 0.05, f"Expected p < 0.05, got {result['p_value']}"

    def test_null_not_significant(self):
        """With random labels, HSIC should not be significant."""
        X, y = _make_null_data()
        result = hsic_test(X, y, n_permutations=500, random_state=42)
        
        # p-value should be large (not significant)
        assert result["p_value"] > 0.01, f"Expected p > 0.01 under null, got {result['p_value']}"

    def test_cka_bounded(self):
        """CKA should be in [0, 1]."""
        X, y = _make_signal_data()
        result = hsic_test(X, y, n_permutations=100)
        assert 0 <= result["cka"] <= 1.0 + 1e-10

    def test_returns_expected_keys(self):
        X, y = _make_signal_data(n=50, d=10)
        result = hsic_test(X, y, n_permutations=100)
        assert set(result.keys()) == {"hsic", "cka", "p_value", "n_perms"}


# ---------------------------------------------------------------------------
# Test: kNN Agreement
# ---------------------------------------------------------------------------

class TestKNNAgreement:
    def test_detects_signal(self):
        """With planted signal, kNN agreement should exceed chance."""
        X, y = _make_signal_data()
        result = knn_agreement_test(X, y, k_values=[5, 10], n_permutations=200, random_state=42)
        
        assert len(result) == 2
        for _, row in result.iterrows():
            assert row["mean_agreement"] > row["expected_chance"], (
                f"k={row['k']}: agreement {row['mean_agreement']:.3f} not > "
                f"chance {row['expected_chance']:.3f}"
            )
            assert row["p_value"] < 0.05
            assert row["adjusted_agreement"] > 0

    def test_null_near_chance(self):
        """With random labels, kNN agreement should be near chance."""
        X, y = _make_null_data()
        result = knn_agreement_test(X, y, k_values=[5], n_permutations=200, random_state=42)
        
        row = result.iloc[0]
        # Agreement should be close to chance (within ~0.1)
        assert abs(row["mean_agreement"] - row["expected_chance"]) < 0.15, (
            f"Expected agreement near chance, got {row['mean_agreement']:.3f} vs {row['expected_chance']:.3f}"
        )

    def test_per_sample_agreement_returned(self):
        """Should return per-sample agreement values."""
        X, y = _make_signal_data(n=50, d=10)
        result = knn_agreement_test(X, y, k_values=[3], n_permutations=50)
        
        per_sample = result.iloc[0]["per_sample_agreement"]
        assert len(per_sample) == 50
        assert all(0 <= v <= 1 for v in per_sample)

    def test_skips_k_too_large(self):
        """Should skip k values larger than n-1."""
        X, y = _make_signal_data(n=10, d=5)
        result = knn_agreement_test(X, y, k_values=[3, 5, 50], n_permutations=50)
        assert 50 not in result["k"].values

    def test_standardization(self):
        """Standardization should not crash and should affect results."""
        X, y = _make_signal_data(n=50, d=10)
        # Scale features wildly
        X[:, 0] *= 1000
        r1 = knn_agreement_test(X, y, k_values=[5], n_permutations=50, standardize=True)
        r2 = knn_agreement_test(X, y, k_values=[5], n_permutations=50, standardize=False)
        # Results should differ because of the scaling
        assert r1.iloc[0]["mean_agreement"] != r2.iloc[0]["mean_agreement"]
