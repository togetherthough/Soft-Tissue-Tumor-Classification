#!/bin/bash
#SBATCH --job-name=exp1_test3ep
#SBATCH --output=logs/exp1_test_%j.log
#SBATCH --error=logs/exp1_test_error_%j.log
#SBATCH --partition=long
#SBATCH --time=0-06:00:00    # Shorter time for quick test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# =============================================================================
# QUICK TEST VERSION - Experiment 1 with 3 Epochs
# =============================================================================
# Quick validation run with only 3 epochs to test everything works
# =============================================================================

echo "=========================================="
echo "EXPERIMENT 1 - QUICK TEST (3 EPOCHS)"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Directory setup
CODE_DIR="${SLURM_SUBMIT_DIR}"
DATA_DIR="/data/scratch/r112276"
CONFIG_FILE="${CODE_DIR}/configs/datasets_cluster.yaml"
RESULTS_DIR="${CODE_DIR}/results/experiment1_test"  # Different output dir for test

echo ""
echo "Code directory: $CODE_DIR"
echo "Data directory: $DATA_DIR"
echo "Config file: $CONFIG_FILE"
echo "Results directory: $RESULTS_DIR"

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

# Threading optimization for GPU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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
"

# Run experiment with 3 EPOCHS
echo ""
echo "=========================================="
echo "Running Experiment 1 - TEST MODE (3 epochs)"
echo "=========================================="

# Force unbuffered output for Python
export PYTHONUNBUFFERED=1

python -u cluster_scripts/experiments/exp1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --epochs-3d 3

# Alternative test configurations:
# With ROI cropping:
#   --use-roi-crop --roi-margin 10 --roi-target-size 128
# With lesion filtering:
#   --filter-preset recommended

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Test completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# Results summary
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ TEST SUCCESS: Results saved to ${RESULTS_DIR}/"
    echo ""
    
    SUMMARY_CSV="${RESULTS_DIR}/combined_benchmarks_summary.csv"
    AVG_CSV="${RESULTS_DIR}/average_scores_per_method.csv"
    
    if [ -f "$SUMMARY_CSV" ]; then
        echo "=========================================="
        echo "Test Results:"
        echo "=========================================="
        head -20 $SUMMARY_CSV
        echo ""
    fi
    
    if [ -f "$AVG_CSV" ]; then
        echo "=========================================="
        echo "⭐ AVERAGE SCORES (3 epochs test):"
        echo "=========================================="
        cat $AVG_CSV
        echo ""
    fi
    
    echo ""
    echo "✅ Validation successful! Ready for full run with 20 epochs."
    echo "To run full experiment: sbatch cluster_scripts/slurm/slurm_exp1.sh"
    echo ""
else
    echo ""
    echo "❌ TEST FAILED: Check error log"
    echo ""
fi

exit $EXIT_CODE
