# HieraCascade Quick Reference

## Training Commands

### Multi-class Classification (Diagnosis)
```bash
# Windows
run_hieracascade.bat 0 Diagnosis

# Linux/Mac
bash run_hieracascade.sh 0 Diagnosis

# Direct Python
python -m hieracascade.quick_start \
    --label_column Diagnosis \
    --fold 0
```

### Binary Classification (Diagnosis_binary)
```bash
# Windows
run_hieracascade.bat 0 Diagnosis_binary

# Linux/Mac
bash run_hieracascade.sh 0 Diagnosis_binary

# Direct Python
python -m hieracascade.quick_start \
    --label_column Diagnosis_binary \
    --fold 0
```

## Notebook Usage

```python
from hieracascade.dataio import create_index_from_sheet

# Load with specific label column
index = create_index_from_sheet(
    data_root='data',
    sheet_path='data/sheet.csv',
    label_column='Diagnosis',  # or 'Diagnosis_binary'
    output_csv='labels.csv'
)

print(f"Loaded {len(index)} studies")
print(f"Categories: {set([item['category'] for item in index])}")
```

## Stratified Sampling

Stratified sampling is **enabled by default** and prints:

```
Fold 0: Using stratified sampling within training sites
  Train size: 120, Val site: hospital_a (30 samples)
  Class distribution in train: {0: 25, 1: 35, 2: 20, 3: 40}
```

To disable (not recommended):
```python
from hieracascade.dataio import create_site_held_out_splits

splits = create_site_held_out_splits(index, stratified=False)
```

## File Locations

- **Configuration Guide**: `hieracascade/LABEL_SELECTION_GUIDE.md`
- **Main Documentation**: `hieracascade/README.md`
- **Tutorial**: `hieracascade/TUTORIAL.md`
- **Demo Notebook**: `notebooks/HieraCascade-Demo.ipynb`

## Common Issues

**Wrong column name?**
```
ValueError: Column 'target' not found in sheet.csv.
Available columns: ['case_id', 'Diagnosis', 'Diagnosis_binary', ...]
```
→ Use `--label_column Diagnosis` or `--label_column Diagnosis_binary`

**Class not recognized?**
```
Warning: Unknown category 'other_tumor' for study XYZ, skipping
```
→ Update `FINE_TO_IDX` in `hieracascade/dataio/utils.py`
