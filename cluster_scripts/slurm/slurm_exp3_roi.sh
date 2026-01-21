#!/bin/bash
#SBATCH --job-name=exp3_clf_roi
#SBATCH --partition=long
#SBATCH --output=logs/exp3_roi_%j.log
#SBATCH --error=logs/exp3_roi_error_%j.log
#SBATCH --nodes=1
# Tip: export SBATCH_NODELIST=gpuXYZ before submission if you must target a specific node.
#SBATCH --time=1-00:00:00    # ADJUST TIME: for all 6 datasets, may need more
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# =============================================================================
# Experiment 3: Classification Head with ROI Cropping
# =============================================================================
# Tests if SAM-Med3D features are discriminative for tumor classification
# using ROI-cropped tumor-centered volumes.
#
# ROI cropping focuses on the tumor region, reducing background noise and
# improving feature extraction for small lesions.
#
# Usage:
#   sbatch cluster_scripts/slurm/slurm_exp3_roi.sh
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Path configuration
CODE_DIR="${SLURM_SUBMIT_DIR}"
RESULTS_DIR="${CODE_DIR}/results/classification_head_roi"
CONFIG_FILE="${CODE_DIR}/configs/datasets_cluster.yaml"

echo ""
echo "Code directory: $CODE_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Config file: $CONFIG_FILE"
echo ""
echo "🎯 ROI CROPPING MODE: Tumor-centered volumes"
echo "   roi_margin=10 voxels, roi_target_size=128"

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
echo "Starting Experiment 3 with ROI Cropping"
echo "=========================================="
echo ""
echo "Preprocessing: ROI CROPPING (tumor-centered)"
echo "  Margin: 10 voxels"
echo "  Target size: 128x128x128"
echo "Model loading: MedIM (recommended)"
echo ""

python cluster_scripts/experiments/exp3_classifier.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --epochs 20 \
    --batch-size 4 \
    --freeze-encoder \
    --use-roi-crop \
    --roi-margin 10 \
    --roi-target-size 128

# Alternative: Combine with lesion filtering
# python cluster_scripts/experiments/exp3_classifier.py \
#     --config ${CONFIG_FILE} \
#     --output-dir ${RESULTS_DIR} \
#     --epochs 20 \
#     --batch-size 4 \
#     --freeze-encoder \
#     --use-roi-crop \
#     --roi-margin 10 \
#     --roi-target-size 128 \
#     --filter-preset recommended \
#     --use-medim

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Experiment completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# Results summary
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: ROI-cropped results saved to ${RESULTS_DIR}/"
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
    echo "  ${CODE_DIR}/logs/exp3_roi_error_${SLURM_JOB_ID}.log"
    echo ""
fi

exit $EXIT_CODE
