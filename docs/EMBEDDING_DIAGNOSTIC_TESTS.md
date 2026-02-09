# Embedding Diagnostic Tests — Experiment Plan

> **Goal:** Before any classifier touches the data, answer a fundamental question:
> *Do SAM-Med3D embeddings actually carry predictive signal about the clinical labels?*

We design three complementary statistical tests at increasing levels of granularity.

---

## Motivation

Our current pipeline is: **SAM-Med3D encoder → pooling → PCA → TabPFN**.
When classification performance is poor, it's unclear *where* the bottleneck lies — is the encoder not encoding relevant anatomy, is pooling destroying signal, or is the classifier underpowered?

These diagnostic tests operate **directly on pooled features** (before any classifier), isolating the representation quality question from everything downstream. Mann-Whitney runs on raw 1,920-dim features; HSIC and kNN run on PCA-reduced features (≤500-dim) to denoise the kernel/distance computations.

---

## The Three Tests

| Level | Test | Question | Null Hypothesis (H₀) |
|-------|------|----------|----------------------|
| **Feature** | Mann-Whitney U | Are individual embedding dimensions diagnostic? | Dimension *d* has the same distribution for class 0 and class 1 |
| **Global** | HSIC | Does the full representation contain label info? | The embedding matrix **X** and label vector **y** are statistically independent |
| **Local** | kNN agreement | Can similar patients help classify new ones? | Nearest-neighbor labels are no better than chance |

### Why these three and not just one?

- **Mann-Whitney** is *univariate* — it can detect signal even if only a handful of dimensions are discriminative, but it misses multivariate interactions.
- **HSIC** is *global and multivariate* — it captures any (including nonlinear) dependence between the full feature matrix and labels, but gives no spatial/local intuition.
- **kNN agreement** is *local* — it directly tests the premise that TabPFN relies on: that nearby points in feature space should share labels. It's also the most interpretable for clinicians ("patients with similar imaging features tend to have the same diagnosis").

Together they cover the **univariate → multivariate → geometric** spectrum.

---

## Experiment Design

### Scope & Isolation

This is a **standalone diagnostic experiment** — it should:
- Live in a single notebook: `notebooks/experiments/Embedding_Diagnostics.ipynb`
- Export results to: `results/embedding_diagnostics/`
- Require **no classifier training** — pure statistical tests on pre-extracted features
- Run on **all 6 datasets** with **ROI-cropped preprocessing only**
- Use **percentile pooling** (1,920-dim features)

### Input Data

For each dataset we need:
1. **Pooled feature matrix** `X` — shape `(n_samples, 1920)` from percentile pooling on ROI-cropped volumes
2. **Label vector** `y` — shape `(n_samples,)`, binary {0, 1}

These are already produced by the existing pipeline's `load_pooled_features()` and `load_labels_from_sheet()` functions. We reuse them directly.

### Feature Spaces

| Test | Input | Rationale |
|------|-------|-----------|
| **Mann-Whitney** | Raw pooled features (1,920-dim) | Rank-based test, no curse of dimensionality, needs all original dimensions |
| **HSIC / CKA** | PCA-reduced (≤500-dim) | Denoises the kernel matrix, removes redundant dimensions |
| **kNN agreement** | PCA-reduced (≤500-dim) | Euclidean distance is more meaningful in lower dimensions |
| **UMAP / t-SNE** | PCA-reduced (≤50-dim) | Standard practice for 2D projection |

### Important: No Train/Test Split Needed

These are **descriptive statistical tests on the full dataset**, not predictive evaluations. We use all available samples for maximum statistical power. The tests answer "is there signal?" not "can we generalize?" — the latter is already answered by our cross-validated classification experiments.

---

## Test 1: Mann-Whitney U (Feature-Level)

### What it does
For each of the 1,920 embedding dimensions independently, test whether the distribution of values differs between class 0 and class 1.

### Procedure
1. For each dimension $d \in \{1, \ldots, 1920\}$:
   - Extract values $x_d^{(0)}$ for class 0, $x_d^{(1)}$ for class 1
   - Compute Mann-Whitney U statistic and p-value
2. Apply **Benjamini-Hochberg FDR correction** at $\alpha = 0.05$ across all 1,920 tests
3. Report:
   - Number of significant dimensions (before and after correction)
   - Distribution of p-values (histogram)
   - Effect sizes (rank-biserial correlation $r = 1 - 2U/(n_0 \cdot n_1)$)
   - Top-k most discriminative dimensions

### Outputs
- **Table:** per-dataset count of significant dimensions at FDR < 0.05
- **Figure 1a:** P-value histogram per dataset (enrichment near 0 = signal, uniform = null)
- **Figure 1b:** Volcano plot (effect size vs. −log₁₀ p-value) per dataset
- **Figure 1c:** Heatmap of top-20 most significant dimensions across datasets (do the same dims light up → generalizable features, or dataset-specific?)

### Interpretation Guide
- **Many significant dims (>100):** Strong, distributed signal — representation is highly informative
- **Few significant dims (10-50):** Sparse signal — a few dimensions are doing the heavy lifting
- **No significant dims:** Either no signal, or signal is multivariate (check HSIC)
- **Same dims across datasets:** SAM-Med3D has learned generalizable anatomical features
- **Different dims per dataset:** Task-specific encoding in different subspaces

---

## Test 2: HSIC (Global Dependence)

### What it does
The Hilbert-Schmidt Independence Criterion measures statistical dependence between two random variables in reproducing kernel Hilbert spaces. Unlike correlation, it captures **any** form of dependence (linear, nonlinear, multivariate interactions).

### Procedure
1. Compute kernel matrices:
   - $K_X$: RBF kernel on feature matrix $X$, bandwidth = median pairwise distance (median heuristic)
   - $K_Y$: Delta kernel on labels $y$ (i.e., $K_Y[i,j] = \mathbb{1}[y_i = y_j]$)
2. Compute HSIC statistic (biased estimator is fine for our sample sizes)
3. **Permutation test** for p-value:
   - Shuffle labels 10,000 times
   - Recompute HSIC each time
   - p-value = fraction of permuted HSIC ≥ observed HSIC
4. Also compute **normalized HSIC** (CKA — Centered Kernel Alignment) for cross-dataset comparability:
   $\text{CKA}(X, Y) = \frac{\text{HSIC}(X, Y)}{\sqrt{\text{HSIC}(X, X) \cdot \text{HSIC}(Y, Y)}}$

### Outputs
- **Table:** per-dataset HSIC value, CKA value, p-value
- **Figure 2a:** Bar chart of CKA scores across datasets (with significance markers)
- **Figure 2b:** Null distribution histogram with observed HSIC marked (one per dataset)

### Interpretation Guide
- **CKA > 0.1, p < 0.05:** Meaningful global dependence — the representation encodes label-relevant information
- **CKA ≈ 0, p > 0.05:** No detectable dependence — features are not informative (or sample size too small)
- **High CKA but poor classification:** Signal exists but classifier can't exploit it (e.g., nonlinear manifold that TabPFN can't capture)
- **Low CKA but good classification:** Sparse signal that HSIC's global view averages out — check Mann-Whitney results

### Computational Notes
- Input: **PCA-reduced features** (≤500 components) — denoises the RBF kernel and makes bandwidth selection more stable
- HSIC requires computing an $n \times n$ kernel matrix — trivial for our dataset sizes (max 246 samples)
- The 10,000 permutations are embarrassingly parallel — each is just a matrix trace, ~1 ms total per permutation at n=246

---

## Test 3: kNN Agreement (Local Geometry)

### What it does
For each sample, find its $k$ nearest neighbors in feature space and check what fraction share the same label. If the embedding is good, nearby points should tend to have the same class.

### Procedure
1. Compute pairwise distances (Euclidean) on **standardized** features (zero-mean, unit-variance per dimension)
2. For $k \in \{3, 5, 7, 10, 15, 20\}$:
   - For each sample $i$, find $k$ nearest neighbors
   - Compute **label agreement** = fraction of neighbors with same label as $i$
   - Average across all samples → **mean kNN agreement**
3. **Permutation baseline:**
   - Shuffle labels 1,000 times
   - Recompute mean kNN agreement each time
   - Expected agreement under null = class prior (i.e., proportion of majority class)
4. Compute **p-value** = fraction of permuted agreements ≥ observed agreement
5. Also compute **adjusted agreement** = (observed − expected) / (1 − expected), analogous to Cohen's kappa

### Outputs
- **Table:** per-dataset, per-k: mean agreement, expected agreement, adjusted agreement, p-value
- **Figure 3a:** Line plot of kNN agreement vs. k for each dataset (with chance-level band)
- **Figure 3b:** Per-sample agreement heatmap → identify "easy" vs. "hard" cases (cases where all neighbors disagree)

### Interpretation Guide
- **Agreement >> chance, stable across k:** Strong local structure — TabPFN/kNN-based methods should work well
- **Agreement ≈ chance:** No local label coherence — either no signal or signal is global/nonlinear
- **Agreement drops sharply with k:** Tight class-specific clusters but classes are interleaved at larger scales
- **High variance across samples:** Some cases are well-separated, others are ambiguous — investigate the hard cases clinically

---

## Test 4: UMAP / t-SNE Visualization (Qualitative)

### What it does
2D projection of the PCA-reduced embedding space, colored by label. Not a statistical test — a visual sanity check that complements the quantitative tests above.

### Procedure
1. PCA-reduce to 50 components (standard pre-step for UMAP/t-SNE)
2. Run **UMAP** (n_neighbors=15, min_dist=0.1) and **t-SNE** (perplexity=30) → 2D
3. Scatter plot colored by class label

### Outputs
- **Figure 4:** 6-panel grid (one per dataset), each showing UMAP 2D projection colored by label
- **Figure 5:** Same grid with t-SNE (for robustness — if both agree, the structure is real)

### Interpretation
- Clear cluster separation → strong signal, consistent with high kNN agreement
- Overlapping clouds → weak or nonlinear signal
- Sub-clusters within a class → potential confounders (scanner, site, etc.)

---

## Compute Time Estimate

All operations are CPU-only, no GPU needed. Largest dataset: GIST (n=246).

| Operation | Per-dataset cost | 6 datasets total | Notes |
|-----------|-----------------|-------------------|-------|
| Load features + labels | ~1 s | ~6 s | I/O bound, .pt files |
| **Mann-Whitney** (1,920 tests) | < 1 s | ~3 s | `scipy.stats.mannwhitneyu` is vectorizable |
| FDR correction | < 0.1 s | < 1 s | Single array operation |
| PCA (1,920 → ≤500) | < 1 s | ~3 s | `sklearn.decomposition.PCA` |
| **HSIC** (246×246 kernel + 10K perms) | ~2-5 s | ~20 s | Kernel: O(n²), each perm: O(n²) trace |
| **kNN agreement** (6 k-values + 1K perms) | ~1-3 s | ~10 s | `sklearn.neighbors.NearestNeighbors` |
| **UMAP** projection | ~2-5 s | ~20 s | `umap-learn` |
| **t-SNE** projection | ~1-3 s | ~10 s | `sklearn.manifold.TSNE` |
| Plotting (all figures) | ~5 s | ~5 s | matplotlib/seaborn |
| **Total** | | **~80 s** | |

**Verdict: This is a notebook experiment.** Total runtime is well under 2 minutes. No SLURM script needed — everything runs interactively in a Jupyter notebook with immediate visual feedback.

---

## Implementation Plan

### Delivery: Single Notebook

Everything lives in **one notebook**: `notebooks/experiments/Embedding_Diagnostics.ipynb`

The notebook is self-contained: loads data, runs all tests, produces all figures, and exports a summary CSV.

### Notebook Structure

| Cell Block | Content |
|------------|----------|
| **0. Setup** | Imports, paths, config (dataset list, pooling strategy, k-values) |
| **1. Data Loading** | Loop over 6 datasets: `load_pooled_features()` + `load_labels_from_sheet()` → dict of {dataset: (X, y)} |
| **2. Mann-Whitney** | Run per-dimension tests, FDR correction, build results DataFrame |
| **3. Mann-Whitney Figures** | P-value histograms, volcano plots, cross-dataset heatmap |
| **4. PCA Reduction** | Fit PCA per dataset, store reduced features for HSIC/kNN/UMAP |
| **5. HSIC** | Compute HSIC + CKA + permutation p-value per dataset |
| **6. HSIC Figures** | CKA bar chart, null distribution histograms |
| **7. kNN Agreement** | Compute agreement for k ∈ {3,5,7,10,15,20} + permutation p-values |
| **8. kNN Figures** | Agreement vs. k line plot, per-sample heatmap |
| **9. UMAP / t-SNE** | 2D projections, scatter plots colored by label |
| **10. Summary** | Traffic-light table (🟢🟡🔴), export to CSV |

### Supporting Code

Statistical test functions go in `med3pipe/tabular/diagnostics.py` — importable and testable:
- `mann_whitney_test(X, y)` → DataFrame of per-dimension results
- `hsic_test(X, y, n_permutations=10000)` → dict with HSIC, CKA, p-value
- `knn_agreement_test(X, y, k_values)` → DataFrame of results

Unit tests in `tests/test_diagnostics.py` with synthetic data.

---

## Dependencies

All tests use only standard scientific Python — no new packages needed:

| Package | Used for | Already in environment? |
|---------|----------|------------------------|
| `scipy.stats.mannwhitneyu` | Mann-Whitney U test | ✅ Yes (scipy) |
| `scipy.stats.false_discovery_control` | BH-FDR correction | ✅ Yes (scipy ≥ 1.11) |
| `sklearn.metrics.pairwise` | Kernel matrices, distances | ✅ Yes |
| `numpy` | Permutation tests, array ops | ✅ Yes |
| `pandas` | Result tables | ✅ Yes |
| `matplotlib` / `seaborn` | Visualization | ✅ Yes |

> **Note:** If scipy < 1.11, use `statsmodels.stats.multitest.multipletests` for FDR correction instead.

---

## Expected Outcomes & Decision Matrix

| Scenario | Mann-Whitney | HSIC/CKA | kNN Agreement | Interpretation | Action |
|----------|-------------|----------|---------------|----------------|--------|
| **A** | Many sig. dims | High CKA, p<0.05 | >> chance | ✅ Strong signal at all levels | Current pipeline is sound |
| **B** | Few sig. dims | High CKA, p<0.05 | >> chance | Signal is multivariate/nonlinear | Consider nonlinear dim. reduction (UMAP → TabPFN) |
| **C** | Many sig. dims | Low CKA | ≈ chance | Marginal signal but no geometric structure | kNN-based classifiers will struggle; try SVM/MLP |
| **D** | No sig. dims | Low CKA | ≈ chance | ❌ No detectable signal | Problem is upstream: encoder or pooling |
| **E** | Mixed across datasets | Mixed | Mixed | Dataset-dependent signal quality | Focus resources on high-signal datasets |

---

## File Structure (after implementation)

```
notebooks/experiments/Embedding_Diagnostics.ipynb       # Main notebook (runs everything)
med3pipe/tabular/diagnostics.py                         # Statistical test functions
tests/test_diagnostics.py                               # Unit tests (synthetic data)
results/embedding_diagnostics/                          # Exported results
    mann_whitney_results.csv                            # Per-dimension p-values & effect sizes
    hsic_results.csv                                    # HSIC, CKA, p-values per dataset
    knn_agreement_results.csv                           # Agreement scores per dataset per k
    summary_table.csv                                   # Cross-dataset traffic-light summary
docs/EMBEDDING_DIAGNOSTIC_TESTS.md                      # This document
```

---

## Resolved Design Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Raw vs. PCA features? | **Raw for Mann-Whitney, PCA for HSIC/kNN** | MW is rank-based (no curse of dim.), HSIC/kNN benefit from denoising |
| Which pooling? | **Percentile only** (1,920-dim) | Our default strategy; extend later if results warrant it |
| Which preprocessing? | **ROI-cropped only** | Our best-performing pipeline; no need to re-validate baseline/filtered here |
| Standardize before MW? | **No** | MW is rank-based, invariant to monotone transforms |
| Notebook or SLURM? | **Notebook** | Total compute ~80 seconds across all 6 datasets — no cluster needed |
| UMAP/t-SNE? | **Yes** | Qualitative complement to the three quantitative tests |
