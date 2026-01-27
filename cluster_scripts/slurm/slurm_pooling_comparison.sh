#!/bin/bash
#SBATCH --job-name=pooling_cmp
#SBATCH --partition=long
#SBATCH --output=logs/pooling_compare_%j.log
#SBATCH --error=logs/pooling_compare_error_%j.log
#SBATCH --nodes=1
# Tip: export SBATCH_NODELIST=gpuXYZ before submission if you must target a specific node.
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# =============================================================================
# Pooling Strategy Comparison Experiment
# =============================================================================
# Compares different pooling strategies for SAM-Med3D embeddings:
#
# 1. Average Pooling (avg):
#    - Global average across all spatial dimensions
#    - Feature dimension: C (e.g., 384)
#
# 2. Multiscale Pooling (multiscale):
#    - Concatenation at 1x1x1, 2x2x2, 4x4x4 scales
#    - Feature dimension: C * 73 (e.g., 28,032)
#
# 3. Percentile Pooling (percentile):
#    - 10th, 25th, 50th, 75th, 90th percentiles per channel
#    - Feature dimension: C * 5 (e.g., 1,920)
#
# All experiments use ROI-cropped volumes and TabPFN with k-fold CV.
#
# Usage:
#   sbatch cluster_scripts/slurm/slurm_pooling_comparison.sh
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Path configuration
CODE_DIR="${SLURM_SUBMIT_DIR}"
RESULTS_DIR="${CODE_DIR}/results/pooling_comparison"
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
echo "Starting Pooling Strategy Comparison"
echo "=========================================="
echo ""
echo "Pooling strategies to compare:"
echo "  1. avg (Global Average Pooling)"
echo "  2. multiscale (1x1x1 + 2x2x2 + 4x4x4 pyramid)"
echo "  3. percentile (10th, 25th, 50th, 75th, 90th)"
echo ""

# Force unbuffered output
export PYTHONUNBUFFERED=1

python -u cluster_scripts/experiments/compare_pooling_strategies.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --roi-margin 30 \
    --n-splits 5

# Options:
#   --datasets gist lipo              : Run only specific datasets
#   --pooling-strategies avg multiscale : Run only specific pooling strategies
#                                         (choices: avg, multiscale, percentile)
#   --roi-margin 30                   : ROI margin in voxels (default: 30)
#   --img-size 128                    : Image size (default: 128)
#   --n-splits 5                      : K-fold splits (default: 5)
#   --n-components-max 500            : Max PCA components (default: 500)
#
# Examples:
#   # Run only average pooling on GIST dataset:
#   python -u cluster_scripts/experiments/compare_pooling_strategies.py \
#       --config ${CONFIG_FILE} --output-dir ${RESULTS_DIR} \
#       --datasets gist --pooling-strategies avg
#
#   # Compare only multiscale and percentile:
#   python -u cluster_scripts/experiments/compare_pooling_strategies.py \
#       --config ${CONFIG_FILE} --output-dir ${RESULTS_DIR} \
#       --pooling-strategies multiscale percentile

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
    
    LATEST_CSV="${RESULTS_DIR}/pooling_comparison_latest.csv"
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
    echo "  ${CODE_DIR}/logs/pooling_compare_error_${SLURM_JOB_ID}.log"
    echo ""
fi

exit $EXIT_CODE
