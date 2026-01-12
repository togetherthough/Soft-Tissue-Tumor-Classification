#!/bin/bash
# =============================================================================
# Diagnose Experiment 3 Issues
# =============================================================================
# This script checks common issues that cause Experiment 3 to fail
#
# Usage:
#   bash cluster_scripts/diagnose_experiment3.sh
# =============================================================================

echo "=========================================="
echo "Experiment 3 Diagnostic Check"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1

echo "Repository root: $REPO_ROOT"
echo ""

# Check 1: Config file
echo "[1/7] Checking config file..."
CONFIG_FILE="configs/datasets_cluster.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo "✅ Config file exists: $CONFIG_FILE"
    
    # Check for duplicate datasets
    DUPLICATES=$(grep "^  [a-z_]*:$" "$CONFIG_FILE" | sort | uniq -d)
    if [ -n "$DUPLICATES" ]; then
        echo "❌ ERROR: Duplicate datasets found in config:"
        echo "$DUPLICATES"
    else
        echo "✅ No duplicate datasets found"
    fi
    
    # List datasets
    echo "   Datasets configured:"
    grep "^  [a-z_]*:$" "$CONFIG_FILE" | sed 's/://g' | sed 's/^  /   - /'
else
    echo "❌ Config file not found: $CONFIG_FILE"
fi
echo ""

# Check 2: Sheet.csv
echo "[2/7] Checking sheet.csv..."
SHEET_CSV="/data/scratch/r112276/sheet.csv"
if [ -f "$SHEET_CSV" ]; then
    echo "✅ sheet.csv found: $SHEET_CSV"
    echo "   Dataset names in sheet.csv:"
    cut -d',' -f1 "$SHEET_CSV" | tail -n +2 | sort | uniq -c | sed 's/^/   /'
else
    echo "❌ sheet.csv not found at: $SHEET_CSV"
    echo "   Checking local sheet.csv..."
    if [ -f "sheet.csv" ]; then
        echo "✅ Found local sheet.csv"
        echo "   Dataset names:"
        cut -d',' -f1 "sheet.csv" | tail -n +2 | sort | uniq -c | sed 's/^/   /'
    else
        echo "❌ No sheet.csv found"
    fi
fi
echo ""

# Check 3: SAM-Med3D checkpoint
echo "[3/7] Checking SAM-Med3D checkpoint..."
CHECKPOINT_PATHS=(
    "SAM-Med3D-main/SAM-Med3D-main/ckpt/sam_med3d_turbo.pth"
    "SAM-Med3D-main/SAM-Med3D-main/ckpt/SAM-Med3D-turbo.pth"
)

CHECKPOINT_FOUND=0
for CKPT in "${CHECKPOINT_PATHS[@]}"; do
    if [ -f "$CKPT" ]; then
        echo "✅ Checkpoint found: $CKPT"
        FILE_SIZE=$(ls -lh "$CKPT" | awk '{print $5}')
        echo "   Size: $FILE_SIZE"
        
        # Check if file is readable
        if [ -r "$CKPT" ]; then
            echo "✅ Checkpoint is readable"
            
            # Check file type
            FILE_TYPE=$(file "$CKPT" 2>/dev/null || echo "file command not available")
            echo "   Type: $FILE_TYPE"
        else
            echo "❌ Checkpoint is not readable"
        fi
        
        CHECKPOINT_FOUND=1
        break
    fi
done

if [ $CHECKPOINT_FOUND -eq 0 ]; then
    echo "❌ No checkpoint found. Tried:"
    for CKPT in "${CHECKPOINT_PATHS[@]}"; do
        echo "   - $CKPT"
    done
    echo ""
    echo "   Download with:"
    echo "   cd SAM-Med3D-main/SAM-Med3D-main/ckpt/"
    echo "   wget https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth"
fi
echo ""

# Check 4: Dataset directories
echo "[4/7] Checking dataset directories..."
DATA_ROOT="/data/scratch/r112276"
if [ -d "$DATA_ROOT" ]; then
    echo "✅ Data root exists: $DATA_ROOT"
    echo "   Datasets found:"
    for ds in crlm desmoid gist lipo liver melanoma; do
        if [ -d "$DATA_ROOT/$ds" ]; then
            CASE_COUNT=$(find "$DATA_ROOT/$ds" -maxdepth 1 -type d -name "*_CT" -o -name "*_MR" 2>/dev/null | wc -l)
            echo "   ✅ $ds ($CASE_COUNT cases)"
        else
            echo "   ❌ $ds (not found)"
        fi
    done
else
    echo "❌ Data root not found: $DATA_ROOT"
    echo "   Running on cluster?"
fi
echo ""

# Check 5: Python environment
echo "[5/7] Checking Python environment..."
if command -v python &> /dev/null; then
    echo "✅ Python found: $(which python)"
    echo "   Version: $(python --version 2>&1)"
    
    # Check PyTorch
    if python -c "import torch" 2>/dev/null; then
        echo "✅ PyTorch installed"
        python -c "import torch; print(f'   Version: {torch.__version__}')"
        python -c "import torch; print(f'   CUDA available: {torch.cuda.is_available()}')" 2>/dev/null
    else
        echo "❌ PyTorch not installed"
    fi
    
    # Check med3pipe
    if python -c "import med3pipe" 2>/dev/null; then
        echo "✅ med3pipe importable"
    else
        echo "❌ med3pipe not importable"
    fi
else
    echo "❌ Python not found in PATH"
fi
echo ""

# Check 6: GPU availability
echo "[6/7] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi available"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/   /'
else
    echo "❌ nvidia-smi not available (running on login node?)"
fi
echo ""

# Check 7: Recent logs
echo "[7/7] Checking recent experiment logs..."
if [ -d "logs" ]; then
    echo "✅ logs/ directory exists"
    RECENT_LOGS=$(find logs -name "exp3*.log" -type f -mtime -7 2>/dev/null | sort -r | head -n 3)
    if [ -n "$RECENT_LOGS" ]; then
        echo "   Recent experiment 3 logs:"
        echo "$RECENT_LOGS" | sed 's/^/   /'
        echo ""
        echo "   Check logs with:"
        echo "   tail -n 50 logs/exp3_error_JOBID.log"
    else
        echo "   No recent experiment 3 logs found"
    fi
else
    echo "⚠️  logs/ directory does not exist (will be created on first run)"
fi
echo ""

# Summary
echo "=========================================="
echo "Diagnostic Summary"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Fix any ❌ issues above"
echo "2. Test with single dataset: python cluster_scripts/run_experiment3_classification_head.py --datasets gist --epochs 3"
echo "3. Submit full job: sbatch cluster_scripts/slurm_experiment3_quick.sh"
echo ""
