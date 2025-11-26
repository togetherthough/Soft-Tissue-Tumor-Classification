#!/bin/bash
#SBATCH --job-name=exp3_clf_filtered
#SBATCH --partition=long
#SBATCH --output=logs/exp3_filtered_%j.log
#SBATCH --error=logs/exp3_filtered_error_%j.log
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
# Experiment 3: Classification Head with Lesion Size Filtering
# =============================================================================
# Tests classification head performance using only high-quality lesions
# (voxels >= 500, dimension >= 5, density >= 0.3)
#
# Results will be saved in a separate folder from unfiltered experiments:
#   results/classification_head/DATASET_filtered_v500_d5_ρ0.30/
#
# Usage:
#   sbatch cluster_scripts/slurm_experiment3_filtered.sh
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
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
echo ""
echo "⚠️  FILTERED MODE: Using recommended lesion size filtering"
echo "   min_voxels=500, min_dimension=5, min_density=0.3"
echo "   Results will be saved in separate '_filtered_*' folders"

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

# Run experiment with filtering
echo ""
echo "=========================================="
echo "Starting Experiment 3 with Filtering"
echo "=========================================="

python cluster_scripts/run_experiment3_classification_head.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --epochs 20 \
    --batch-size 4 \
    --freeze-encoder \
    --filter-preset recommended

# Alternative: Use custom filtering parameters instead of preset
# Uncomment and modify these lines to use custom thresholds:
#    --min-voxels 500 \
#    --min-dimension 5 \
#    --min-density 0.3

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Experiment completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# Results summary
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: Filtered results saved to ${RESULTS_DIR}/"
    echo ""
    echo "Note: Results are in dataset-specific filtered folders:"
    echo "  e.g., gist_filtered_v500_d5_ρ0.30/"
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
    echo "  ${CODE_DIR}/logs/exp3_filtered_error_${SLURM_JOB_ID}.log"
    echo ""
fi

exit $EXIT_CODE
