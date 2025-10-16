# Utility Scripts

Collection of utility scripts for the project.

---

## Scripts

### `analyze_labels.py`
Analyze the dataset structure and label distribution from `sheet.csv`.

**Usage**:
```bash
python scripts/analyze_labels.py
```

**Output**: Prints dataset statistics and label distributions.

---

### `test_sam_features.py`
Evaluate SAM-Med3D feature quality quickly with a small classification head.

**Usage**:
```bash
python scripts/test_sam_features.py --dataset gist --epochs 10 --freeze
```

See `docs/SAM_FEATURE_EVALUATION.md` for details.

---

### `update_nb_paths.py` (legacy)
Internal path-fix utility for notebooks. Safe to remove if not used.

---

### `cleanup_docs.ps1` (one-off)
Deletes redundant docs per the minimal set. Run once, then remove.

---

## Adding New Scripts

Place utility scripts here to keep the project root clean.

**Back to main**: [../README.md](../README.md)
