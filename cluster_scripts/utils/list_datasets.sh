#!/bin/bash
# Script to discover datasets and generate config entries

echo "=========================================="
echo "Dataset Discovery Tool"
echo "=========================================="
echo ""

DATA_DIR="/data/scratch/r112276"
SHEET_CSV="${DATA_DIR}/sheet.csv"

echo "Searching for datasets in: $DATA_DIR"
echo ""

# Find all directories in data folder
DATASETS=$(find "$DATA_DIR" -maxdepth 1 -type d -not -name "r112276" | sort)

if [ -z "$DATASETS" ]; then
    echo "No datasets found in $DATA_DIR"
    exit 1
fi

echo "Found datasets:"
echo "----------------------------------------"
for ds_path in $DATASETS; do
    ds_name=$(basename "$ds_path")
    
    # Count .nii or .nii.gz files
    nii_count=$(find "$ds_path" -maxdepth 1 -type f \( -name "*.nii" -o -name "*.nii.gz" \) 2>/dev/null | wc -l)
    
    echo "  - $ds_name ($nii_count .nii/.nii.gz files)"
done
echo ""

echo "=========================================="
echo "Generating Config Entries"
echo "=========================================="
echo ""
echo "Add these to configs/datasets_cluster.yaml:"
echo ""
echo "datasets:"

for ds_path in $DATASETS; do
    ds_name=$(basename "$ds_path")
    ds_upper=$(echo "$ds_name" | tr '[:lower:]' '[:upper:]')
    
    # Try to determine if CT or MR by looking at file names
    mr_files=$(find "$ds_path" -maxdepth 1 -type f -name "*MR*" -o -name "*mr*" 2>/dev/null | wc -l)
    if [ $mr_files -gt 0 ]; then
        case_suffix="_MR"
    else
        case_suffix="_CT"
    fi
    
    cat << EOF
  $ds_name:
    dataset_root: /data/scratch/r112276/$ds_name
    category: $ds_name
    ct_name: ct_${ds_upper}
    labels:
      sheet_csv: /data/scratch/r112276/sheet.csv
      dataset_name: ${ds_upper}  # Check sheet.csv for exact name
      subject_col: Subject
      label_col: Diagnosis_binary
      case_suffix: ${case_suffix}
    prepare:
      case_glob: null
      max_cases: null
    split:
      ratio: 0.8
      seed: 2025
    extraction:
      img_size: 128

EOF
done

echo ""
echo "=========================================="
echo "Verify Dataset Names in sheet.csv"
echo "=========================================="

if [ -f "$SHEET_CSV" ]; then
    echo ""
    echo "Unique dataset names in your sheet.csv:"
    if command -v python &> /dev/null; then
        python << 'PYEOF'
import pandas as pd
try:
    df = pd.read_csv('/data/scratch/r112276/sheet.csv')
    if 'Dataset' in df.columns:
        datasets = df['Dataset'].unique()
        for ds in sorted(datasets):
            print(f"  - {ds}")
    else:
        print("  Warning: No 'Dataset' column found in sheet.csv")
        print("  Available columns:", list(df.columns))
except Exception as e:
    print(f"  Error reading sheet.csv: {e}")
PYEOF
    else
        echo "  Python not available - check manually"
    fi
else
    echo "  sheet.csv not found at: $SHEET_CSV"
fi

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo "1. Review the generated config entries above"
echo "2. Verify dataset names match your sheet.csv"
echo "3. Copy the entries to configs/datasets_cluster.yaml"
echo "4. Upload the updated config to the cluster"
echo "5. Run: sbatch cluster_scripts/slurm_train_and_test.sh"
echo ""
