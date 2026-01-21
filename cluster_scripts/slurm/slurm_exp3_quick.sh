#!/bin/bash
#SBATCH --job-name=exp3_quick
#SBATCH --output=logs/exp3_quick_%j.log
#SBATCH --error=logs/exp3_quick_error_%j.log
#SBATCH --partition=short         # Use short partition for quick tests
#SBATCH --time=0-02:00:00         # 2 hours should be enough for 5 epochs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# =============================================================================
# Experiment 3: Quick Test with Fewer Epochs
# =============================================================================
# Fast version for testing/debugging with only 5 epochs per dataset
# Perfect for initial validation before full run
#
# Usage:
#   sbatch cluster_scripts/slurm/slurm_exp3_quick.sh
#
# To test single dataset:
#   sbatch cluster_scripts/slurm/slurm_exp3_quick.sh gist
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Path configuration
CODE_DIR="${SLURM_SUBMIT_DIR}"
RESULTS_DIR="${CODE_DIR}/results/classification_head_quick"
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
module load Python/3.9
module load CUDA/12.3.0

# Activate virtual environment (thesis_peron)
source /trinity/home/r112276/Med3Tab-PFN/thesis_peron/bin/activate

# Set threading environment variables
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
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

# Verify checkpoint exists
echo ""
echo "Checking SAM-Med3D checkpoint..."
CHECKPOINT_PATH="${CODE_DIR}/SAM-Med3D-main/SAM-Med3D-main/ckpt/sam_med3d_turbo.pth"
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "✅ Checkpoint found: $CHECKPOINT_PATH"
    ls -lh "$CHECKPOINT_PATH"
else
    echo "⚠️  WARNING: Checkpoint not found at: $CHECKPOINT_PATH"
    echo "   Download from: https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth"
fi

# Determine datasets to run
if [ $# -eq 0 ]; then
    # Run all datasets
    DATASETS_ARG=""
    echo "Running ALL datasets with 5 epochs each"
else
    # Run specific datasets passed as arguments
    DATASETS_ARG="--datasets $@"
    echo "Running ONLY datasets: $@"
fi

# Run experiment
echo ""
echo "=========================================="
echo "Starting Experiment 3 (Quick - 5 epochs)"
echo "=========================================="

python cluster_scripts/experiments/exp3_classifier.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --epochs 5 \
    --batch-size 4 \
    --freeze-encoder \
    ${DATASETS_ARG}

# Alternative test configurations:
# With ROI cropping:
#   --use-roi-crop --roi-margin 10 --roi-target-size 128
# With lesion filtering:
#   --filter-preset recommended

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
        echo "Interpretation:"
        echo "  AUC > 0.70: ✅ Good features, proceed to full pipeline"
        echo "  AUC 0.60-0.70: 🟡 Moderate, consider fine-tuning"
        echo "  AUC < 0.60: ❌ Poor features, investigate data/model"
    fi
else
    echo ""
    echo "❌ FAILED: Check error log at:"
    echo "  ${CODE_DIR}/logs/exp3_quick_error_${SLURM_JOB_ID}.log"
    echo ""
fi

exit $EXIT_CODE
