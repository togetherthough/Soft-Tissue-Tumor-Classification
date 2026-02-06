# Pooling Strategy System

## Overview

The Med3Tab-PFN pipeline now supports three different pooling strategies for SAM-Med3D embeddings. You can select the pooling strategy via a command-line flag (`--pooling-strategy`) across all experiments.

## Available Pooling Strategies

### 1. Average Pooling (avg)
- **Feature Dimension**: C (e.g., 384)
- **Description**: Global average pooling across all spatial dimensions
- **Use Case**: Fast baseline, good for most datasets
- **Characteristics**:
  - Simplest and fastest
  - Captures overall feature activation
  - Works well when entire tumor context is important

### 2. Multiscale Pooling (multiscale)
- **Feature Dimension**: C × 73 (e.g., 28,032)
- **Description**: Spatial pyramid pooling at 1×1×1, 2×2×2, and 4×4×4 scales
- **Use Case**: When spatial structure at multiple scales is important
- **Characteristics**:
  - Captures multi-scale spatial information
  - High-dimensional features (requires PCA)
  - Computation: 73 = 1³ + 2³ + 4³ = 1 + 8 + 64
  - Best for heterogeneous tumors with complex spatial patterns

### 3. Percentile Pooling (percentile) - DEFAULT
- **Feature Dimension**: C × 5 (e.g., 1,920)
- **Description**: 10th, 25th, 50th, 75th, 90th percentiles per channel
- **Use Case**: When distribution of feature activations is important
- **Characteristics**:
  - Captures statistical distribution of features
  - Robust to outliers
  - Good for tumors with heterogeneous intensity patterns

## Usage Across Experiments

### Experiment 1: Benchmarks (exp1_benchmarks.py)

```bash
# Default (percentile pooling)
python cluster_scripts/experiments/exp1_benchmarks.py \
    --config configs/datasets.yaml

# Multiscale pooling
python cluster_scripts/experiments/exp1_benchmarks.py \
    --config configs/datasets.yaml \
    --pooling-strategy multiscale

# Percentile pooling
python cluster_scripts/experiments/exp1_benchmarks.py \
    --config configs/datasets.yaml \
    --pooling-strategy percentile
```

### Preprocessing Comparison (compare_preprocessing.py)

```bash
# Default (percentile pooling)
python cluster_scripts/experiments/compare_preprocessing.py \
    --config configs/datasets.yaml

# Compare preprocessing strategies with multiscale pooling
python cluster_scripts/experiments/compare_preprocessing.py \
    --config configs/datasets.yaml \
    --pooling-strategy multiscale

# Compare preprocessing strategies with percentile pooling
python cluster_scripts/experiments/compare_preprocessing.py \
    --config configs/datasets.yaml \
    --pooling-strategy percentile
```

### Pooling Comparison (compare_pooling_strategies.py)

```bash
# Compare all strategies (default behavior)
python cluster_scripts/experiments/compare_pooling_strategies.py \
    --config configs/datasets.yaml

# Compare only multiscale and percentile
python cluster_scripts/experiments/compare_pooling_strategies.py \
    --config configs/datasets.yaml \
    --pooling-strategies multiscale percentile

# Test single strategy on specific dataset
python cluster_scripts/experiments/compare_pooling_strategies.py \
    --config configs/datasets.yaml \
    --datasets gist \
    --pooling-strategies avg
```

## Implementation Details

### Core Changes

1. **med3pipe/sam/core.py**
   - Added `pool_avg()`, `pool_multiscale()`, `pool_percentile()` functions
   - Modified `average_pool_embedding()` to accept `pooling_strategy` parameter
   - Modified `load_pooled_features()` to accept `pooling_strategy` parameter
   - Added `POOLING_STRATEGIES` dictionary for strategy lookup

2. **med3pipe/tabular/stratify.py**
   - Added `pooling_strategy` parameter to `stratified_kfold_features()`
   - Added `pooling_strategy` parameter to `stratified_features_split()`
   - Passes `pooling_strategy` to `load_pooled_features()` calls

3. **med3pipe/pipelines/end_to_end.py**
   - Added `pooling_strategy` parameter to `run_single_dataset()`
   - Passes `pooling_strategy` to `stratified_features_split()`

4. **med3pipe/pipelines/multi_dataset.py**
   - Added `pooling_strategy` parameter to `_run_multi_core()`
   - Added `pooling_strategy` parameter to `run_multi_tabpfn()`
   - Added `pooling_strategy` parameter to `run_multi_localpfn()`
   - Passes `pooling_strategy` through entire call chain

5. **cluster_scripts/experiments/exp1_benchmarks.py**
   - Added `--pooling-strategy` command-line argument
   - Added `pooling_strategy` parameter to `run_experiment()`
   - Passes `pooling_strategy` to `run_multi_tabpfn()` and `run_multi_localpfn()` calls

6. **cluster_scripts/experiments/compare_pooling_strategies.py**
   - Added `--pooling-strategies` command-line argument (can specify multiple)
   - Filters which strategies to run based on argument
   - Default behavior: runs all strategies if no filter specified

7. **cluster_scripts/experiments/compare_preprocessing.py**
   - Added `--pooling-strategy` command-line argument
   - Added `pooling_strategy` parameter to all experiment functions
   - Passes `pooling_strategy` to `run_multi_tabpfn()` calls for each preprocessing variant

### Experiments NOT Requiring Pooling Strategy

**exp3_classifier.py and compare_classifier_preprocessing.py**:
- These use `run_classification_head_experiment()` which trains an end-to-end classifier
- They don't use pooling+TabPFN pipeline, so pooling strategy doesn't apply
- The classification head has its own global average pooling layer built-in

## Technical Considerations

### Feature Dimensions
- **Average**: 384 features (C dimensions)
- **Multiscale**: 28,032 features (384 × 73)
- **Percentile**: 1,920 features (384 × 5)

### Memory Requirements
- **Average**: Lowest memory usage
- **Multiscale**: Highest memory usage (requires PCA for TabPFN compatibility)
- **Percentile**: Moderate memory usage

### Computation Time
- **Average**: Fastest (~1x baseline)
- **Percentile**: Moderate (~1.2x baseline)
- **Multiscale**: Slowest (~1.5-2x baseline due to PCA)

### PCA Application
TabPFN has a feature limit (~500-1000). For high-dimensional strategies:
- **Multiscale** (28,032 dims): Always requires PCA
- **Percentile** (1,920 dims): May require PCA depending on sample size
- **Average** (384 dims): Usually doesn't require PCA

## Default Pooling Strategy

The default pooling strategy is `'percentile'` (percentile pooling), based on experimental results showing superior performance:
- Existing scripts without `--pooling-strategy` flag will use percentile pooling
- All function signatures have `pooling_strategy='percentile'` as default
- Provides best balance of performance and computational efficiency

## Best Practices

### When to Use Each Strategy

**Average Pooling** (recommended starting point):
- Quick experiments and baselines
- When computational resources are limited
- When tumor morphology is relatively homogeneous

**Multiscale Pooling**:
- Complex tumors with heterogeneous patterns
- When spatial structure at multiple scales matters
- When you have sufficient computational resources
- Research exploring multi-scale representations

**Percentile Pooling**:
- Tumors with high intensity variability
- When robust statistical features are desired
- Middle ground between avg and multiscale in terms of dimensionality

### Experimental Workflow

1. **Start with average pooling** as baseline:
   ```bash
   python cluster_scripts/experiments/exp1_benchmarks.py --pooling-strategy avg
   ```

2. **Test other strategies** if results aren't satisfactory:
   ```bash
   python cluster_scripts/experiments/compare_pooling_strategies.py \
       --datasets <your_dataset>
   ```

3. **Analyze results** using the visualization notebook:
   ```bash
   # notebooks/visualization/Pooling_Comparison_Results.ipynb
   ```

4. **Run full experiments** with best-performing strategy:
   ```bash
   python cluster_scripts/experiments/exp1_benchmarks.py \
       --pooling-strategy <best_strategy>
   ```

## SLURM Integration

### For exp1 (benchmarks):
```bash
# Edit cluster_scripts/slurm/slurm_exp1.sh
python -u cluster_scripts/experiments/exp1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --pooling-strategy multiscale  # Add this line
```

### For pooling comparison:
```bash
# Edit cluster_scripts/slurm/slurm_pooling_comparison.sh
python -u cluster_scripts/experiments/compare_pooling_strategies.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --pooling-strategies avg multiscale  # Compare specific strategies
```

### For preprocessing comparison:
```bash
# Edit cluster_scripts/slurm/slurm_preprocessing_comparison.sh
python -u cluster_scripts/experiments/compare_preprocessing.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --pooling-strategy multiscale  # Add this line
```

## Troubleshooting

### Error: "Invalid pooling strategy"
- **Cause**: Typo in strategy name
- **Solution**: Use exactly `'avg'`, `'multiscale'`, or `'percentile'`

### Memory Error with Multiscale
- **Cause**: High-dimensional features (28,032) exceed memory
- **Solution**: 
  1. Ensure PCA is enabled (should be automatic)
  2. Reduce `n_components_max` parameter
  3. Use fewer datasets or smaller batch size
  4. Consider using `percentile` instead

### Inconsistent Results Across Strategies
- **Cause**: Different feature dimensions affect model capacity
- **Solution**: This is expected behavior. Document which strategy was used for each result.

### Embeddings from Different Strategies Mix
- **Cause**: Reusing embeddings without re-extraction
- **Solution**: 
  - Embeddings are strategy-agnostic (stored as spatial feature maps)
  - Pooling is applied when loading features, not during extraction
  - Safe to change strategies without re-extracting embeddings

## Related Files

- [POOLING_STRATEGY_GUIDE.md](../cluster_scripts/experiments/POOLING_STRATEGY_GUIDE.md) - Detailed usage examples
- [Pooling_Comparison_Results.ipynb](../notebooks/visualization/Pooling_Comparison_Results.ipynb) - Visualization notebook
- [compare_pooling_strategies.py](../cluster_scripts/experiments/compare_pooling_strategies.py) - Comparison experiment
- [exp1_benchmarks.py](../cluster_scripts/experiments/exp1_benchmarks.py) - Main benchmark experiment
- [compare_preprocessing.py](../cluster_scripts/experiments/compare_preprocessing.py) - Preprocessing comparison experiment

## Summary of Affected Experiments

| Experiment Script | Pooling Strategy Support | Notes |
|-------------------|-------------------------|-------|
| `exp1_benchmarks.py` | ✅ Yes | `--pooling-strategy` flag |
| `compare_pooling_strategies.py` | ✅ Yes | `--pooling-strategies` flag (multiple) |
| `compare_preprocessing.py` | ✅ Yes | `--pooling-strategy` flag |
| `exp3_classifier.py` | ❌ No | Uses end-to-end classification head |
| `compare_classifier_preprocessing.py` | ❌ No | Uses end-to-end classification head |

**Rule of thumb**: If the experiment uses TabPFN or LoCalPFN (via `run_multi_tabpfn` or `run_multi_localpfn`), it supports pooling strategy selection. If it uses `run_classification_head_experiment`, it doesn't (has built-in GAP).

## Citation

If you use the multiscale or percentile pooling strategies in your research, please consider citing:
- Spatial Pyramid Pooling (He et al., 2015) for multiscale pooling
- Our methodology paper (when published) for the integrated system
