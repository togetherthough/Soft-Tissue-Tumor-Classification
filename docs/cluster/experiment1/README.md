# Experiment 1: Head-to-Head Benchmarks

Complete benchmark comparison of all methods on tumor classification datasets.

## Methods Compared

1. **Med3-TabPFN** - TabPFN on SAM-Med3D features
2. **Med3-LoCalPFN** - Local context-aware PFN variant
3. **DenseNet121-3D** - 3D convolutional baseline
4. **ViT-3D** - 3D Vision Transformer baseline

## Quick Start

### Using Cluster Scripts

**Without filtering (baseline):**
```bash
sbatch cluster_scripts/slurm_train_and_test.sh
```

**With filtering (recommended):**
```bash
sbatch cluster_scripts/slurm_train_and_test_filtered.sh
```

**Compare both:**
```bash
sbatch cluster_scripts/slurm_train_and_test.sh          # Unfiltered
sbatch cluster_scripts/slurm_train_and_test_filtered.sh # Filtered
# Results in separate folders - easy to compare!
```

### Using Python

```bash
python cluster_scripts/run_experiment1_benchmarks.py \
    --config configs/datasets_cluster.yaml \
    --output-dir results/experiment1
```

## Configuration Options

```bash
# Run specific datasets only
python cluster_scripts/run_experiment1_benchmarks.py \
    --datasets gist lipo \
    --config configs/datasets.yaml

# Skip certain methods
python cluster_scripts/run_experiment1_benchmarks.py \
    --skip-tabpfn \
    --skip-localpfn \
    --skip-baselines

# Adjust 3D model training epochs
python cluster_scripts/run_experiment1_benchmarks.py \
    --epochs-3d 20

# With lesion size filtering (NEW)
python cluster_scripts/run_experiment1_benchmarks.py \
    --config configs/datasets.yaml \
    --filter-preset recommended

# Custom filtering
python cluster_scripts/run_experiment1_benchmarks.py \
    --config configs/datasets.yaml \
    --min-voxels 500 \
    --min-dimension 5 \
    --min-density 0.3
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to datasets config YAML | `configs/datasets.yaml` |
| `--output-dir` | Output directory for results | `results/experiment1` |
| `--datasets` | Specific datasets to run | All in config |
| `--skip-tabpfn` | Skip TabPFN method | False |
| `--skip-localpfn` | Skip LoCalPFN method | False |
| `--skip-baselines` | Skip 3D baselines | False |
| `--epochs-3d` | Epochs for 3D models | 4 |
| `--min-voxels` | Minimum voxel count (filtering) | None |
| `--min-dimension` | Minimum dimension (filtering) | None |
| `--min-density` | Minimum density (filtering) | None |
| `--filter-preset` | Use preset: recommended, conservative, lenient | None |

## Output Files

Results are saved in `results/experiment1/`:

```
results/experiment1/
├── combined_benchmarks_summary.csv    # Main results table
├── tabpfn_gist/                       # TabPFN outputs per dataset
│   ├── metrics.json
│   ├── predictions.csv
│   └── ...
├── localpfn_gist/                     # LoCalPFN outputs
├── densenet121_gist/                  # DenseNet outputs
└── vit3d_gist/                        # ViT-3D outputs
```

## Notes

- **GPU recommended** for 3D baselines (DenseNet, ViT)
- TabPFN and LoCalPFN are faster, run on CPU or GPU
- Results are timestamped and organized by method/dataset
- Summary CSV combines all methods for easy comparison
- **Filtering consistency**: When filtering is enabled, ALL methods (TabPFN, LoCalPFN, DenseNet, ViT) use the SAME filtered dataset for fair comparison

## See Also

- **Experiment 3**: [../experiment3/README.md](../experiment3/README.md) - Classification head experiments
- **Cluster Guide**: [../README.md](../README.md) - General cluster usage
- **Filtering**: [../../FILTERING.md](../../FILTERING.md) - Lesion size filtering
