#!/bin/bash
#SBATCH --job-name=clf_preproc_cmp
#SBATCH --partition=long
#SBATCH --output=logs/clf_compare_%j.log
#SBATCH --error=logs/clf_compare_error_%j.log
#SBATCH --nodes=1
# Tip: export SBATCH_NODELIST=gpuXYZ before submission if you must target a specific node.
#SBATCH --time=2-00:00:00    # 2 days - runs 3 experiments per dataset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# =============================================================================
# Classification Head: Three-Way Preprocessing Comparison
# =============================================================================
# Compares performance of three preprocessing approaches for classification head:
# 1. Baseline: Full-volume, no lesion filtering
# 2. Filtered Baseline: Full-volume with lesion size filtering
# 3. ROI-Cropped: Adaptive crop/pad (tumor-centered volumes)
#
# Usage:
#   sbatch cluster_scripts/slurm/slurm_compare_classifier.sh
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Path configuration
CODE_DIR="${SLURM_SUBMIT_DIR}"
RESULTS_DIR="${CODE_DIR}/results/classification_head_comparison"
CONFIG_FILE="${CODE_DIR}/configs/datasets_cluster.yaml"

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
# Using 1 thread prevents CPU contention when GPU does the heavy lifting
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Let SLURM manage GPU assignment (it sets CUDA_VISIBLE_DEVICES automatically)
# Don't override unless you have a specific reason
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
echo "Starting Classification Head Preprocessing Comparison"
echo "=========================================="
echo ""
echo "This experiment will run 3 preprocessing variants per dataset:"
echo "  1. Baseline (full-volume, no filtering)"
echo "  2. Filtered Baseline (full-volume + lesion filtering)"
echo "  3. ROI-Cropped (adaptive crop/pad)"
echo ""
echo "Expected runtime: ~2-3 hours per dataset (depends on dataset size)"
echo "=========================================="

python cluster_scripts/experiments/compare_classifier_preprocessing.py

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
    
    COMPARISON_CSV="${RESULTS_DIR}/preprocessing_comparison.csv"
    ALL_RESULTS_CSV="${RESULTS_DIR}/all_results.csv"
    
    if [ -f "$COMPARISON_CSV" ]; then
        echo "=========================================="
        echo "Preprocessing Comparison Summary:"
        echo "=========================================="
        cat $COMPARISON_CSV
        echo ""
    fi
    
    if [ -f "$ALL_RESULTS_CSV" ]; then
        echo "=========================================="
        echo "All Results:"
        echo "=========================================="
        cat $ALL_RESULTS_CSV
        echo ""
    fi
    
    echo "=========================================="
    echo "Output files:"
    echo "  - ${COMPARISON_CSV}"
    echo "  - ${ALL_RESULTS_CSV}"
    echo "  - Individual results in: ${RESULTS_DIR}/[experiment_name]/[dataset]/"
    echo "=========================================="
else
    echo ""
    echo "❌ FAILED: Check error log at:"
    echo "  ${CODE_DIR}/logs/clf_compare_error_${SLURM_JOB_ID}.log"
    echo ""
fi

exit $EXIT_CODE
