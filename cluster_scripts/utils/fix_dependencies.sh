#!/bin/bash
# Quick fix script for missing dependencies and files

echo "=========================================="
echo "Fixing Dependencies and File Locations"
echo "=========================================="

# Activate conda environment (adjust this to your environment name)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis_peron  # CHANGE THIS to your actual environment name

echo ""
echo "1. Installing missing packages..."
pip install torchvision tabpfn

echo ""
echo "2. Verifying installation..."
python -c "
import torch
import torchvision
import tabpfn
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ torchvision: {torchvision.__version__}')
print(f'✓ tabpfn: {tabpfn.__version__}')
"

echo ""
echo "3. Setting up shared sheet.csv file..."

# Find where sheet.csv exists
echo "Searching for sheet.csv..."
SHEET_FOUND=$(find /trinity/home/r112276/Med3Tab-PFN -name "sheet.csv" -type f 2>/dev/null | head -1)

if [ -z "$SHEET_FOUND" ]; then
    echo "⚠️  WARNING: sheet.csv not found in repo!"
    echo "   Please locate your sheet.csv file and copy it to:"
    echo "   /data/scratch/r112276/sheet.csv"
else
    echo "✓ Found sheet.csv at: $SHEET_FOUND"
    
    echo ""
    echo "4. Copying sheet.csv to shared location..."
    
    # Create directory if it doesn't exist
    mkdir -p /data/scratch/r112276
    
    # Copy sheet.csv to shared location (one file for all datasets!)
    cp "$SHEET_FOUND" /data/scratch/r112276/sheet.csv
    
    echo "✓ Copied to /data/scratch/r112276/sheet.csv"
fi

echo ""
echo "5. Verifying shared sheet.csv exists..."
if [ -f "/data/scratch/r112276/sheet.csv" ]; then
    echo "✓ /data/scratch/r112276/sheet.csv exists"
    echo "  All datasets will use this single file!"
else
    echo "✗ /data/scratch/r112276/sheet.csv NOT found"
    echo "  Please copy your sheet.csv to this location"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload the updated configs/datasets_cluster.yaml to the cluster"
echo "2. Resubmit your job: sbatch cluster_scripts/slurm_train_and_test.sh"
echo ""
