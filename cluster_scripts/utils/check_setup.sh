#!/bin/bash
# =============================================================================
# Cluster Setup Verification Script
# =============================================================================
# Run this script on the cluster to verify your environment is ready
# Usage: bash check_setup.sh

echo "=========================================="
echo "Med3Tab-PFN Cluster Setup Verification"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SUCCESS=0
WARNINGS=0
ERRORS=0

# -----------------------------------------------------------------------------
# Check 1: Directory Structure
# -----------------------------------------------------------------------------
echo "1. Checking directory structure..."

# Auto-detect code directory (where this script is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CODE_DIR="$(dirname "$SCRIPT_DIR")"  # Go up one level from cluster_scripts/
DATA_DIR="${CODE_DIR}"  # Assume data is in the same repo by default

if [ -d "$CODE_DIR" ]; then
    echo -e "   ${GREEN}✓${NC} Code directory exists: $CODE_DIR"
    ((SUCCESS++))
else
    echo -e "   ${RED}✗${NC} Code directory NOT found: $CODE_DIR"
    ((ERRORS++))
fi

if [ -d "$DATA_DIR" ]; then
    echo -e "   ${GREEN}✓${NC} Data directory exists: $DATA_DIR"
    ((SUCCESS++))
else
    echo -e "   ${RED}✗${NC} Data directory NOT found: $DATA_DIR"
    ((ERRORS++))
fi

# Check key subdirectories
if [ -d "$CODE_DIR/med3pipe" ]; then
    echo -e "   ${GREEN}✓${NC} med3pipe package found"
    ((SUCCESS++))
else
    echo -e "   ${RED}✗${NC} med3pipe package NOT found"
    ((ERRORS++))
fi

if [ -d "$CODE_DIR/sam-med3d" ]; then
    echo -e "   ${GREEN}✓${NC} SAM-Med3D directory found"
    ((SUCCESS++))
else
    echo -e "   ${YELLOW}⚠${NC} SAM-Med3D directory NOT found"
    ((WARNINGS++))
fi

# Create necessary directories
mkdir -p ${CODE_DIR}/logs
mkdir -p ${CODE_DIR}/results/experiment1

echo ""

# -----------------------------------------------------------------------------
# Check 2: Python Environment
# -----------------------------------------------------------------------------
echo "2. Checking Python environment..."

if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo -e "   ${GREEN}✓${NC} Python found: $PYTHON_VERSION"
    ((SUCCESS++))
else
    echo -e "   ${RED}✗${NC} Python NOT found"
    ((ERRORS++))
fi

# Check if conda is available
if command -v conda &> /dev/null; then
    echo -e "   ${GREEN}✓${NC} Conda is available"
    CONDA_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
    if [ -n "$CONDA_ENV" ]; then
        echo -e "   ${GREEN}✓${NC} Active conda environment: $CONDA_ENV"
    else
        echo -e "   ${YELLOW}⚠${NC} No conda environment activated"
        ((WARNINGS++))
    fi
    ((SUCCESS++))
else
    echo -e "   ${YELLOW}⚠${NC} Conda not available (using system Python)"
    ((WARNINGS++))
fi

echo ""

# -----------------------------------------------------------------------------
# Check 3: Python Packages
# -----------------------------------------------------------------------------
echo "3. Checking required Python packages..."

REQUIRED_PACKAGES=("torch" "numpy" "pandas" "scikit-learn" "yaml" "nibabel" "SimpleITK")

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        VERSION=$(python -c "import $pkg; print($pkg.__version__ if hasattr($pkg, '__version__') else 'unknown')" 2>/dev/null)
        echo -e "   ${GREEN}✓${NC} $pkg ($VERSION)"
        ((SUCCESS++))
    else
        echo -e "   ${RED}✗${NC} $pkg NOT installed"
        ((ERRORS++))
    fi
done

echo ""

# -----------------------------------------------------------------------------
# Check 4: PyTorch CUDA Support
# -----------------------------------------------------------------------------
echo "4. Checking PyTorch and CUDA..."

if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "   ${GREEN}✓${NC} PyTorch version: $TORCH_VERSION"
    
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')")
        echo -e "   ${GREEN}✓${NC} CUDA available: version $CUDA_VERSION"
        echo -e "   ${GREEN}✓${NC} GPU count: $GPU_COUNT"
        echo -e "   ${GREEN}✓${NC} GPU 0: $GPU_NAME"
        ((SUCCESS+=3))
    else
        echo -e "   ${RED}✗${NC} CUDA NOT available (GPU acceleration disabled)"
        echo -e "   ${YELLOW}⚠${NC} This is OK for login node, but required on compute nodes"
        ((WARNINGS++))
    fi
else
    echo -e "   ${RED}✗${NC} PyTorch not installed"
    ((ERRORS++))
fi

echo ""

# -----------------------------------------------------------------------------
# Check 5: Data Files
# -----------------------------------------------------------------------------
echo "5. Checking data files..."

# Check for datasets from config
if [ -f "$CODE_DIR/configs/datasets.yaml" ]; then
    # Try to parse dataset names from YAML
    DATASETS=$(grep -E '^\s+[a-z]+:' "$CODE_DIR/configs/datasets.yaml" | sed 's/://g' | tr -d ' ')
    echo -e "   Datasets in config: $(echo $DATASETS | tr '\n' ' ')"
    
    for ds in $DATASETS; do
        # Check if dataset directory exists
        if [ -d "$CODE_DIR/data/$ds" ] || [ -d "$CODE_DIR/$ds" ]; then
            DS_DIR="$CODE_DIR/data/$ds"
            [ ! -d "$DS_DIR" ] && DS_DIR="$CODE_DIR/$ds"
            echo -e "   ${GREEN}✓${NC} $ds dataset directory found: $DS_DIR"
            
            # Check for sheet.csv
            if [ -f "$DS_DIR/sheet.csv" ] || [ -f "$CODE_DIR/sheet.csv" ]; then
                echo -e "   ${GREEN}✓${NC} $ds sheet.csv found"
                ((SUCCESS++))
            else
                echo -e "   ${YELLOW}⚠${NC} $ds sheet.csv not found at expected location"
                ((WARNINGS++))
            fi
            ((SUCCESS++))
        else
            echo -e "   ${YELLOW}⚠${NC} $ds dataset directory not found (will be checked at runtime)"
            ((WARNINGS++))
        fi
    done
else
    echo -e "   ${YELLOW}⚠${NC} Cannot check datasets - config file not found"
    ((WARNINGS++))
fi

echo ""

# -----------------------------------------------------------------------------
# Check 6: SAM-Med3D Checkpoint
# -----------------------------------------------------------------------------
echo "6. Checking SAM-Med3D checkpoint..."

CHECKPOINT_FOUND=0
if [ -d "$CODE_DIR/sam-med3d/ckpt" ]; then
    CKPT_DIR="$CODE_DIR/sam-med3d/ckpt"
    echo -e "   ${GREEN}✓${NC} Checkpoint directory found"
    ((SUCCESS++))
    
    if [ -f "$CKPT_DIR/sam_med3d_turbo.pth" ]; then
        CKPT_SIZE=$(du -h "$CKPT_DIR/sam_med3d_turbo.pth" | cut -f1)
        echo -e "   ${GREEN}✓${NC} sam_med3d_turbo.pth found (${CKPT_SIZE})"
        CHECKPOINT_FOUND=1
        ((SUCCESS++))
    elif [ -f "$CKPT_DIR/SAM-Med3D-turbo.pth" ]; then
        CKPT_SIZE=$(du -h "$CKPT_DIR/SAM-Med3D-turbo.pth" | cut -f1)
        echo -e "   ${GREEN}✓${NC} SAM-Med3D-turbo.pth found (${CKPT_SIZE})"
        CHECKPOINT_FOUND=1
        ((SUCCESS++))
    else
        echo -e "   ${RED}✗${NC} SAM-Med3D checkpoint NOT found"
        echo -e "   ${YELLOW}⚠${NC} Download from: https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth"
        echo -e "   ${YELLOW}⚠${NC} Save to: $CKPT_DIR/sam_med3d_turbo.pth"
        ((ERRORS++))
    fi
else
    echo -e "   ${RED}✗${NC} SAM-Med3D checkpoint directory NOT found"
    echo -e "   ${YELLOW}⚠${NC} Create: mkdir -p $CODE_DIR/sam-med3d/ckpt"
    ((ERRORS++))
fi

echo ""

# -----------------------------------------------------------------------------
# Check 7: SLURM Availability
# -----------------------------------------------------------------------------
echo "7. Checking SLURM..."

if command -v sbatch &> /dev/null; then
    echo -e "   ${GREEN}✓${NC} SLURM is available (sbatch found)"
    ((SUCCESS++))
    
    # Check SLURM partitions
    PARTITIONS=$(sinfo -h -o "%P" | tr '\n' ', ')
    echo -e "   ${GREEN}✓${NC} Available partitions: $PARTITIONS"
    ((SUCCESS++))
else
    echo -e "   ${RED}✗${NC} SLURM NOT found (are you on the cluster?)"
    ((ERRORS++))
fi

echo ""

# -----------------------------------------------------------------------------
# Check 8: Script Files
# -----------------------------------------------------------------------------
echo "8. Checking experiment scripts..."

if [ -f "$CODE_DIR/cluster_scripts/run_experiment1_benchmarks.py" ]; then
    echo -e "   ${GREEN}✓${NC} Python experiment script found"
    ((SUCCESS++))
else
    echo -e "   ${RED}✗${NC} cluster_scripts/run_experiment1_benchmarks.py NOT found"
    ((ERRORS++))
fi

if [ -f "$CODE_DIR/cluster_scripts/slurm_train_and_test.sh" ]; then
    echo -e "   ${GREEN}✓${NC} SLURM batch script found (slurm_train_and_test.sh)"
    ((SUCCESS++))
elif [ -f "$CODE_DIR/cluster_scripts/slurm_experiment1.sh" ]; then
    echo -e "   ${GREEN}✓${NC} SLURM batch script found (slurm_experiment1.sh)"
    ((SUCCESS++))
else
    echo -e "   ${RED}✗${NC} SLURM batch script NOT found"
    ((ERRORS++))
fi

if [ -f "$CODE_DIR/configs/datasets.yaml" ]; then
    echo -e "   ${GREEN}✓${NC} Dataset config found"
    ((SUCCESS++))
else
    echo -e "   ${RED}✗${NC} configs/datasets.yaml NOT found"
    ((ERRORS++))
fi

echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "${GREEN}Successful checks: $SUCCESS${NC}"
echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
echo -e "${RED}Errors: $ERRORS${NC}"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ Your environment is ready!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review and customize: cluster_scripts/slurm_train_and_test.sh"
    echo "  2. Submit job: sbatch cluster_scripts/slurm_train_and_test.sh"
    echo "  3. Monitor: squeue -u \$USER"
    echo "  4. View logs: tail -f logs/exp1_*.log"
    echo ""
    exit 0
elif [ $ERRORS -le 3 ]; then
    echo -e "${YELLOW}⚠ Your environment has some issues but may still work${NC}"
    echo ""
    echo "Please address the errors above before running the experiment."
    echo ""
    exit 1
else
    echo -e "${RED}✗ Your environment has critical issues${NC}"
    echo ""
    echo "Please fix the errors above before proceeding."
    echo "See cluster_scripts/README.md for setup instructions."
    echo ""
    exit 2
fi
