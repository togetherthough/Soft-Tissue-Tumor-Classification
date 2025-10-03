# HieraCascade Update Summary - Label Selection Feature

**Date**: 2025-10-03  
**Status**: ✅ Complete

## What Was Added

### 1. **Label Column Selection Mechanism**

Added the ability to select different label columns from `sheet.csv` for training.

**Supported Columns:**
- `Diagnosis` - Multi-class tumor classification (default)
- `Diagnosis_binary` - Binary malignant/benign classification

**Where Implemented:**
- `hieracascade/dataio/sheet_loader.py` - Core loading logic
- `hieracascade/quick_start.py` - Command-line interface
- `run_hieracascade.bat` / `run_hieracascade.sh` - Convenience scripts

### 2. **Stratified Sampling Verification**

Enhanced the cross-validation split function with stratified sampling support.

**Features:**
- ✅ Stratified sampling enabled by default
- ✅ Maintains class balance in training sets
- ✅ Prints class distribution for each fold
- ✅ Automatic validation of sufficient samples

**Where Implemented:**
- `hieracascade/dataio/utils.py` - `create_site_held_out_splits()`

### 3. **Comprehensive Documentation**

Created detailed guides for users and examination committees.

**New Documentation:**
- `hieracascade/LABEL_SELECTION_GUIDE.md` - Complete usage guide (10+ examples)
- `hieracascade/QUICK_REFERENCE.md` - Quick command reference
- Updated `hieracascade/README.md` - Added label selection section

## Usage Examples

### Command Line

```bash
# Multi-class classification
python -m hieracascade.quick_start \
    --label_column Diagnosis \
    --data_root data \
    --sheet_csv data/sheet.csv

# Binary classification
python -m hieracascade.quick_start \
    --label_column Diagnosis_binary \
    --data_root data \
    --sheet_csv data/sheet.csv
```

### Convenience Scripts

```bash
# Windows
run_hieracascade.bat 0 Diagnosis
run_hieracascade.bat 0 Diagnosis_binary

# Linux/Mac
bash run_hieracascade.sh 0 Diagnosis
bash run_hieracascade.sh 0 Diagnosis_binary
```

### Python/Notebook

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

## Technical Details

### API Changes

**sheet_loader.py:**
```python
def load_from_sheet_csv(
    sheet_path: str,
    data_root: str,
    label_column: str = 'target',  # NEW PARAMETER
    study_id_col: str = 'case_id',
    modality_col: Optional[str] = 'modality',
    site_col: Optional[str] = 'site'
) -> List[Dict]:
    """Load dataset with configurable label column"""
```

**utils.py:**
```python
def create_site_held_out_splits(
    index: List[Dict],
    n_folds: Optional[int] = None,
    stratified: bool = True  # NEW PARAMETER
) -> List[Tuple[List[Dict], List[Dict]]]:
    """Create CV splits with optional stratification"""
```

### Stratified Sampling Output

When stratified sampling is active, you'll see:

```
Fold 0: Using stratified sampling within training sites
  Train size: 120, Val site: hospital_a (30 samples)
  Class distribution in train: {0: 25, 1: 35, 2: 20, 3: 40}

Found 4 unique categories: ['crlm', 'gist', 'lipo', 'melanoma']
```

This confirms:
- Class balance is maintained
- Each class has sufficient representation
- Site-held-out validation is working

### Error Handling

**Invalid column name:**
```
ValueError: Column 'Target' not found in sheet.csv.
Available columns: ['case_id', 'Diagnosis', 'Diagnosis_binary', 'modality', 'site'].
Use --label_column to specify one of: 'Diagnosis', 'Diagnosis_binary', etc.
```

**Unknown category:**
```
Warning: Unknown category 'sarcoma' for study Study-001, skipping
```

## Files Modified/Created

### Modified Files (7)
1. `hieracascade/dataio/sheet_loader.py` - Added label_column parameter
2. `hieracascade/dataio/utils.py` - Added stratified sampling
3. `hieracascade/quick_start.py` - Added --label_column argument
4. `hieracascade/README.md` - Added label selection section
5. `run_hieracascade.bat` - Added label column support
6. `run_hieracascade.sh` - Added label column support
7. `hieracascade/__init__.py` - Updated docstrings (removed STS)

### New Files (3)
1. `hieracascade/LABEL_SELECTION_GUIDE.md` - Complete usage guide
2. `hieracascade/QUICK_REFERENCE.md` - Quick reference card
3. `HIERACASCADE_UPDATE_SUMMARY.md` - This summary document

## Backwards Compatibility

✅ **Fully backwards compatible**

- Default `label_column='target'` maintains old behavior
- Default `stratified=True` improves splits without breaking changes
- Existing scripts work without modification
- New features are opt-in via command-line arguments

## Testing Checklist

- [x] Multi-class classification loads correctly
- [x] Binary classification loads correctly
- [x] Stratified sampling prints class distribution
- [x] Invalid column names show helpful error messages
- [x] Command-line scripts work with new parameters
- [x] Notebook examples run successfully
- [x] Documentation is comprehensive and clear

## Next Steps for Users

1. **Try both label types:**
   ```bash
   # Compare multi-class vs binary
   run_hieracascade.bat 0 Diagnosis
   run_hieracascade.bat 0 Diagnosis_binary
   ```

2. **Verify stratification:**
   - Check console output for class distribution
   - Ensure balanced representation across folds

3. **Review documentation:**
   - Read `LABEL_SELECTION_GUIDE.md` for examples
   - Check `QUICK_REFERENCE.md` for common commands

4. **Start training:**
   - Use the quick start script for convenience
   - Monitor stratification output to verify splits

## Benefits

### For Multi-class Classification
- Fine-grained tumor type prediction
- Hierarchical coarse/fine predictions
- Better clinical specificity

### For Binary Classification
- Simplified malignant/benign decision
- Higher confidence with fewer classes
- Useful for initial screening

### For Both
- ✅ Stratified sampling ensures balanced training
- ✅ Flexible label selection without code changes
- ✅ Clear documentation for examination committees
- ✅ Production-ready with comprehensive error handling

## Code Quality

- **Student-appropriate comments**: Academic style suitable for thesis
- **No "STS" references**: Changed to "soft tissue tumor"
- **Professional documentation**: Ready for examination committee
- **Comprehensive error messages**: Helpful for debugging
- **Type hints**: All functions properly typed
- **Docstrings**: Complete with examples

## Conclusion

The HieraCascade pipeline now supports flexible label selection from `sheet.csv` with verified stratified sampling. The implementation is:

- ✅ **Production-ready** - Tested and documented
- ✅ **User-friendly** - Simple command-line interface
- ✅ **Well-documented** - Comprehensive guides and examples
- ✅ **Academic-quality** - Suitable for thesis presentation
- ✅ **Backwards-compatible** - No breaking changes

All features are ready for immediate use in training and evaluation workflows.
