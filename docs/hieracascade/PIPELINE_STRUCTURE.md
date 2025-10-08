# Two Pipeline Structure

## Pipeline 1: HieraCascade (Multi-class Tumor Types)

**Task**: Classify tumor types (CRLM, GIST, Desmoid, Lipo, Liver, Melanoma)

**Label Column**: `Dataset`

**Architecture**:
- Stage-1: Coarse families (malignant/benign/other) + saliency
- Stage-2: Fine tumor types (6 classes) + coarse families (3 classes) - HIERARCHICAL
- Uses dual heads: `head_fine` and `head_coarse`

**Notebook**: `hieracascade_multiclass.ipynb`

---

## Pipeline 2: BinaryCascade (Binary Classification)

**Task**: Classify benign vs. malignant

**Label Column**: `Diagnosis_binary`

**Architecture**:
- Stage-1: Binary (benign/malignant) + saliency
- Stage-2: Binary (benign/malignant) with MIL - SIMPLIFIED
- Uses single head: `head`

**Notebook**: `binarycascade_final.ipynb` (currently named `hieracascade_final.ipynb`)

---

## Action Items

1. ✅ Keep current binary Stage-2 model (single head)
2. ⚠️ Create/restore hierarchical Stage-2 model (dual heads)
3. ⚠️ Create two separate training scripts
4. ⚠️ Create two notebooks
