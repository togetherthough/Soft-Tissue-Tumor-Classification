#!/bin/bash
#SBATCH --job-name=preproc_cmp
#SBATCH --partition=long
#SBATCH --output=logs/preprocessing_compare_%j.log
#SBATCH --error=logs/preprocessing_compare_error_%j.log
#SBATCH --nodes=1
# Tip: export SBATCH_NODELIST=gpuXYZ before submission if you must target a specific node.
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# =============================================================================
# Preprocessing Comparison Experiment
# =============================================================================
# Compares three preprocessing strategies for SAM-Med3D embeddings:
#
# 1. Baseline:
#    - Full-volume, no lesion filtering
#    - All samples included regardless of lesion size
#
# 2. Filtered Baseline:
#    - Full-volume with lesion size filtering
#    - Removes samples with lesions too small/sparse
#
# 3. ROI-Cropped:
#    - Adaptive crop/pad (tumor-centered volumes)
#    - Focuses on lesion region with configurable margin
#
# All experiments use TabPFN with k-fold cross-validation.
#
# Usage:
#   sbatch cluster_scripts/slurm/slurm_preprocessing_comparison.sh [pooling_strategy]
#
# Arguments:
#   pooling_strategy: Pooling strategy to use (avg, multiscale, percentile)
#                     Default: avg
#
# Examples:
#   sbatch cluster_scripts/slurm/slurm_preprocessing_comparison.sh
#   sbatch cluster_scripts/slurm/slurm_preprocessing_comparison.sh percentile
#   sbatch cluster_scripts/slurm/slurm_preprocessing_comparison.sh multiscale
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
# Parse arguments
POOLING_STRATEGY=${1:-avg}

echo ""
echo "Configuration:"
echo "  Pooling strategy: $POOLING_STRATEGY"
echo ""
# Path configuration
CODE_DIR="${SLURM_SUBMIT_DIR}"
RESULTS_DIR="${CODE_DIR}/results/preprocessing_comparison"
CONFIG_FILE="${CODE_DIR}/configs/datasets_cluster.yaml"

# Fall back to regular config if cluster config doesn't exist
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="${CODE_DIR}/configs/datasets.yaml"
fi

echo ""
echo "Code directory: $CODE_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Config file: $CONFIG_FILE"

# Create directories
mkdir -p ${CODE_DIR}/logs
mkdir -p ${RESULTS_DIR}

cd ${CODE_DIR} || exit 1

# Environment setup
echo ""
echo "=========================================="
echo "Setting up environment..."
echo "=========================================="

# Load modules
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

# Activate virtual environment (thesis_peron)
source /trinity/home/r112276/Med3Tab-PFN/thesis_peron/bin/activate

# Set threading environment variables for optimal GPU performance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Let SLURM manage GPU assignment
echo "SLURM assigned GPU(s): $CUDA_VISIBLE_DEVICES"

# Verify environment
echo ""
echo "Python version:"
python --version

echo ""
echo "PyTorch and CUDA info:"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

# Run experiment
echo ""
echo "=========================================="
echo "Starting Preprocessing Comparison"
echo "=========================================="
echo ""
echo "Preprocessing strategies to compare:"
echo "  1. Baseline (Full-Volume, No Filtering)"
echo "  2. Filtered Baseline (Full-Volume + Lesion Filtering)"
echo "  3. ROI-Cropped (Adaptive Crop/Pad)"
echo ""

# Force unbuffered output
export PYTHONUNBUFFERED=1

python -u cluster_scripts/experiments/compare_preprocessing.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --roi-margin 30 \
    --roi-target-size 128 \
    --min-voxels 300 \
    --min-dimension 5 \
    --min-density 0.1 \
    --n-splits 5 \
    --n-components-max 500 \
    --pooling-strategy ${POOLING_STRATEGY}

# Options:
#   --datasets gist lipo       : Run only specific datasets
#   --roi-margin 30            : ROI margin in voxels (default: 30)
#   --roi-target-size 128      : ROI target volume size (default: 128)
#   --min-voxels 300           : Minimum lesion voxels (default: 300)
#   --min-dimension 5          : Minimum dimension (default: 5)
#   --min-density 0.1          : Minimum lesion density (default: 0.1)
#   --n-splits 5               : K-fold splits (default: 5)
#   --skip-baseline            : Skip baseline experiment
#   --skip-filtered            : Skip filtered experiment
#   --skip-roi                 : Skip ROI experiment

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Experiment completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# Results summary
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: Results saved to ${RESULTS_DIR}/"
    echo ""
    
    LATEST_CSV="${RESULTS_DIR}/preprocessing_comparison_latest.csv"
    if [ -f "$LATEST_CSV" ]; then
        echo "=========================================="
        echo "Results Summary:"
        echo "=========================================="
        cat $LATEST_CSV
        echo ""
    fi
    
    echo "=========================================="
    echo "Output files:"
    ls -lh ${RESULTS_DIR}/ 2>/dev/null || echo "  (directory exists but empty)"
    echo "=========================================="
else
    echo ""
    echo "❌ FAILED: Check error log at:"
    echo "  ${CODE_DIR}/logs/preprocessing_compare_error_${SLURM_JOB_ID}.log"
    echo ""
fi

exit $EXIT_CODE
