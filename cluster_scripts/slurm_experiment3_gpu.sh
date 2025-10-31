#!/bin/bash
#SBATCH --job-name=exp3_clf_head
#SBATCH --output=logs/exp3_%j.log
#SBATCH --error=logs/exp3_error_%j.log
#SBATCH --partition=gpu      # ← Using GPU partition
#SBATCH --time=1-00:00:00    # 1 day max
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# =============================================================================
# Experiment 3: Classification Head on SAM-Med3D Features (GPU PARTITION)
# =============================================================================
# Tests if SAM-Med3D features are discriminative for tumor classification
# by training a simple classification head on top of the frozen encoder.
#
# Usage:
#   sbatch cluster_scripts/slurm_experiment3_gpu.sh
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start Time: $(date)"
echo "=========================================="

# Path configuration
CODE_DIR="${SLURM_SUBMIT_DIR}"
RESULTS_DIR="${CODE_DIR}/results/classification_head"
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
module load Python/3.10

# Activate virtual environment
source /trinity/home/r112276/Med3Tab-PFN/venv/bin/activate

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

# Run experiment on ALL datasets (no --datasets flag = runs all from config)
echo ""
echo "=========================================="
echo "Starting Experiment 3 - All 6 Datasets"
echo "=========================================="

python cluster_scripts/run_experiment3_classification_head.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --epochs 10 \
    --batch-size 4 \
    --freeze-encoder

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
    
    SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
    if [ -f "$SUMMARY_CSV" ]; then
        echo "=========================================="
        echo "Results Summary:"
        echo "=========================================="
        cat $SUMMARY_CSV
        echo ""
    fi
else
    echo ""
    echo "❌ FAILED: Check error log at:"
    echo "  ${CODE_DIR}/logs/exp3_error_${SLURM_JOB_ID}.log"
    echo ""
fi

exit $EXIT_CODE
