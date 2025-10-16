# Experiment Notebooks

This directory contains two main experiment notebooks for evaluating SAM-Med3D based approaches on tumor classification.

## Experiment 1: Benchmarks (Experiment1-Benchmarks.ipynb)

**Purpose:** Compare PFN-based methods against 3D CNN baselines using **pre-trained** SAM-Med3D features.

**Methods evaluated:**
- **Med3-TabPFN**: TabPFN classifier on SAM-Med3D features (frozen encoder)
- **Med3-LoCalPFN**: Local context-aware PFN with adapter (frozen encoder)
- **DenseNet121-3D**: End-to-end 3D CNN baseline
- **ViT-3D**: End-to-end 3D Vision Transformer baseline

**Key characteristics:**
- Uses pre-trained SAM-Med3D-turbo checkpoint (no fine-tuning)
- Fast to run (~30 minutes for 2 datasets)
- Evaluates zero-shot transfer learning capability
- Feature extraction only, encoder weights are frozen

**When to use:**
- Quick evaluation of pre-trained features
- Limited computational resources
- Testing new classification head architectures
- Baseline comparisons

---

## Experiment 2: Fine-tuned TabPFN/LoCalPFN (Experiment2-Finetuned-TabPFN-LoCalPFN.ipynb)

**Purpose:** Fine-tune SAM-Med3D encoder, then test **TabPFN and LoCalPFN** with fine-tuned features.

**Workflow:**
1. Prepare dataset in SAM-Med3D format (imagesTr/labelsTr)
2. **Fine-tune** SAM-Med3D image encoder on segmentation task (per dataset)
3. Extract features from **fine-tuned** encoder
4. Run **TabPFN** with fine-tuned features
5. Run **LoCalPFN** with fine-tuned features
6. Compare with pre-trained baseline

**Key characteristics:**
- Trains the SAM-Med3D encoder on your specific data
- Uses interactive prompts during fine-tuning (native SAM-Med3D training)
- Tests TabPFN/LoCalPFN with domain-adapted features
- Slow to run (~2-4 hours per dataset with 20 epochs)
- Direct comparison of pre-trained vs fine-tuned features

**When to use:**
- Measure TabPFN/LoCalPFN performance gain from fine-tuning
- Sufficient computational resources (GPU with 12GB+ VRAM)
- Domain-specific data that differs from SAM-Med3D training distribution
- Research experiments on domain adaptation impact

---

## Quick Comparison

| Aspect | Experiment 1 (Benchmarks) | Experiment 2 (Fine-tuned PFN) |
|--------|---------------------------|-------------------------------|
| **Encoder** | Frozen (pre-trained) | Fine-tuned on target data |
| **Methods** | TabPFN, LoCalPFN, DenseNet, ViT | TabPFN, LoCalPFN |
| **Time** | ~30 min for 2 datasets | ~2-4 hours per dataset |
| **GPU** | Optional (CPU works) | Required (CUDA) |
| **Performance** | Good zero-shot transfer | Better domain adaptation |
| **Use case** | Quick baselines | Measure fine-tuning impact |

---

## Configuration Files

Both notebooks use `configs/datasets.yaml` for dataset configuration:

```yaml
datasets:
  gist:
    category: gist
    ct_name: ct_GIST
    dataset_root: gist
    labels:
      sheet_csv: sheet.csv
      dataset_name: GIST
      subject_col: Subject
      label_col: Diagnosis_binary
      case_suffix: _CT
```

---

## Fine-tuning Parameters (Experiment 2)

Key hyperparameters to adjust:

```python
FINETUNE_CONFIG = {
    'num_epochs': 20,        # 2-5 for testing, 20-50 for production
    'batch_size': 4,         # Depends on GPU memory
    'lr': 8e-5,              # Learning rate
    'weight_decay': 0.1,
}
```

**Memory requirements:**
- 12GB GPU: batch_size=2-4
- 24GB GPU: batch_size=4-8
- Multi-GPU: Enable `multi_gpu=True`

---

## Expected Performance Gains

Fine-tuning typically provides:
- **+5-15% accuracy** improvement over frozen features
- **Better segmentation** masks on target domain
- **More robust** features for classification

Trade-off:
- Requires more time and compute
- Risk of overfitting on small datasets
- Need for proper validation to prevent data leakage

---

## Output Structure

### Experiment 1 (Benchmarks)
```
notebooks/
├── tabpfn_runs/
│   └── [dataset]_[ct_name]_[timestamp]/
│       ├── metrics.json
│       └── predictions.csv
├── baselines_3d/
│   └── [dataset]_[ct_name]_[method]/
└── combined_benchmarks_summary.csv
```

### Experiment 2 (Fine-tuned PFN)
```
notebooks/finetuned_pfn_results/
├── [dataset]_finetune_workdir/
│   └── [category]_[ct_name]_ft/
│       ├── sam_model_dice_best.pth
│       ├── sam_model_latest.pth
│       └── Loss.png, Dice.png
├── tabpfn_finetuned/
│   └── [dataset]_[ct_name]_[timestamp]/
│       ├── metrics.json
│       └── predictions.csv
├── localpfn_finetuned/
│   └── [dataset]_[ct_name]_[timestamp]/
│       ├── metrics.json
│       └── predictions.csv
├── finetuned_pfn_results.csv
└── pretrained_vs_finetuned_comparison.csv
```

---

## Troubleshooting

### Experiment 1
- **Issue:** TabPFN out of memory
  - **Solution:** Reduce number of features or use smaller batch size in PFN

### Experiment 2
- **Issue:** CUDA out of memory during fine-tuning
  - **Solution:** Reduce `batch_size` to 2 or 1
- **Issue:** Training very slow
  - **Solution:** Reduce `num_epochs` or use multi-GPU
- **Issue:** Checkpoint not found
  - **Solution:** Check `work_dir/[task_name]/` for available checkpoints

---

## Citation

If you use these experiments in your research, please cite:

```bibtex
@misc{wang2024sammed3d,
  title={SAM-Med3D: Towards General-purpose Segmentation Models for Volumetric Medical Images},
  author={Wang, Haoyu and Guo, Sizheng and Ye, Jin and ...},
  year={2024},
  eprint={2310.15161},
  archivePrefix={arXiv},
}
```
