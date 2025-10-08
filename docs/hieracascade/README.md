# HieraCascade Documentation

Documentation for the cascade deep learning pipelines (BinaryCascade & HieraCascade).

---

## 🌟 Start Here

**[TWO_CASCADES_EXPLAINED.md](TWO_CASCADES_EXPLAINED.md)** - Complete explanation of both pipelines

---

## Quick Reference

### BinaryCascade (Current Priority)
- Binary classification: benign vs. malignant
- Uses `Diagnosis_binary` column
- Single classification head

### HieraCascade (Research)
- Multi-class: tumor types (CRLM, GIST, Desmoid, Lipo, Liver, Melanoma)
- Uses `Dataset` column  
- Hierarchical dual heads (fine + coarse)

---

## Documentation Files

### Essential Guides
1. **[TWO_CASCADES_EXPLAINED.md](TWO_CASCADES_EXPLAINED.md)** - Complete comparison
2. **[COMPLETE_PROJECT_OVERVIEW.md](COMPLETE_PROJECT_OVERVIEW.md)** - All three approaches
3. **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** - Setup summary

### BinaryCascade Specific
4. **[README_BINARY_TASK.md](README_BINARY_TASK.md)** - Quick reference
5. **[BINARY_CLASSIFICATION_SETUP.md](BINARY_CLASSIFICATION_SETUP.md)** - Technical guide
6. **[FINAL_SETUP_SUMMARY.md](FINAL_SETUP_SUMMARY.md)** - Complete setup
7. **[MIGRATION_TO_BINARY.md](MIGRATION_TO_BINARY.md)** - What changed

### Analysis & Context
8. **[LABEL_ANALYSIS.md](LABEL_ANALYSIS.md)** - Dataset structure analysis
9. **[README_BINARY_MIL_QUESTION.md](README_BINARY_MIL_QUESTION.md)** - Design decisions
10. **[DOES_BINARY_MIL_APPLY.md](DOES_BINARY_MIL_APPLY.md)** - Approach comparison
11. **[BINARY_MIL_ANALYSIS_SUMMARY.md](BINARY_MIL_ANALYSIS_SUMMARY.md)** - Executive summary
12. **[ANALYSIS_INDEX.md](ANALYSIS_INDEX.md)** - Analysis file index

### Reference
13. **[HIERACASCADE.md](HIERACASCADE.md)** - Main cascade overview
14. **[PIPELINE_STRUCTURE.md](PIPELINE_STRUCTURE.md)** - Pipeline structure

---

## Training Commands

### BinaryCascade
```bash
python -m hieracascade.quick_start --data_root data --fold 0
```

### HieraCascade
```bash
python -m hieracascade.quick_start_hierarchical --data_root data --fold 0
```

---

## Architecture

Both pipelines use:
- **Stage-1** (Scout): Full volume → predictions + saliency map
- **Stage-2** (Expert): Crops from saliency → MIL → final predictions

**Difference**: BinaryCascade has single binary head, HieraCascade has dual hierarchical heads.

---

## Files Organization

This directory contains all documentation that was moved from the project root for cleaner organization.

**Back to main**: [../../README.md](../../README.md)
