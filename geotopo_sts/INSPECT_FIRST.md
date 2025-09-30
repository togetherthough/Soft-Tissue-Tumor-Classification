# 🔍 First Step: Inspect Your GIST Data Structure

Before running the main pipeline, let's see how your data is actually organized!

## Quick Inspection

Run this in your notebook or terminal:

### Option 1: From Notebook

Add and run this cell in `QuickStart_GIST.ipynb`:

```python
from geotopo_sts.inspect_gist_structure import inspect_directory_structure

# Inspect your GIST data
inspect_directory_structure(r'C:\Users\cahel\Desktop\Med3Tab-PFN\data\gist', max_cases=5)
```

### Option 2: From Terminal

```bash
cd geotopo_sts
python inspect_gist_structure.py --root ../data/gist --max-cases 5
```

---

## What This Shows

The script will display:

1. **📁 All GIST case folders found**
2. **📄 Detailed file structure** for first few cases
3. **📊 File type summary** across all cases
4. **✓ Which NIfTI file patterns exist**
5. **💡 Recommendations** for how to proceed

---

## Expected Output Example

```
📁 Inspecting: C:\Users\cahel\Desktop\Med3Tab-PFN\data\gist
============================================================

✓ Found 246 folders matching 'GIST-*_CT'

📊 Total unique case folders: 246

============================================================
DETAILED STRUCTURE (first 3 cases):
============================================================

1. GIST-001_CT/
   📄 image.nii.gz (45.23 MB)
   📄 mask.nii.gz (0.45 MB)
   📄 metadata.json (0.01 MB)

2. GIST-002_CT/
   📁 NIFTI/
      📄 image.nii.gz (50.12 MB)
      📄 segmentation.nii.gz (0.52 MB)

...
```

---

## Common Scenarios

### Scenario A: Files directly in case folder
```
GIST-001_CT/
├── image.nii.gz
└── mask.nii.gz
```
✓ **The updated loader handles this!**

### Scenario B: Files in NIFTI subfolder
```
GIST-001_CT/
└── NIFTI/
    ├── image.nii.gz
    └── mask.nii.gz
```
✓ **The updated loader handles this too!**

### Scenario C: Different file names
```
GIST-001_CT/
├── ct.nii.gz
└── segmentation.nii.gz
```
✓ **The loader checks for common alternatives!**

### Scenario D: Non-standard names
```
GIST-001_CT/
├── patient_scan.nii.gz
└── tumor_segmentation.nii.gz
```
❌ **You'll need to rename files or update the loader**

---

## Next Steps

### After inspecting:

1. **If files are found**: Continue with the notebook!
   ```python
   cases = discover_gist_cases(gist_root)
   # Should now show: "Found X GIST CT cases"
   ```

2. **If files NOT found**: Check the recommendations from the script
   - Are files actually in NIfTI format?
   - Do they have different names?
   - Update the `discover_gist_cases()` function if needed

3. **If structure is unusual**: Let me know the actual structure and I'll update the loader!

---

## Quick Fix: Update File Search

If your files have different names, update this in `gist_data_loader.py`:

```python
# Around line 54-58, add your file names:
for img_name in ["image.nii.gz", "image.nii", "YOUR_IMAGE_NAME.nii.gz"]:
    ...

# Around line 62-66, add your mask names:
for mask_name in ["mask.nii.gz", "YOUR_MASK_NAME.nii.gz"]:
    ...
```

---

**Run the inspection script first, then come back here with the results!** 🔍
