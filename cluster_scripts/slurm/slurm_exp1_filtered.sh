#!/bin/bash
#SBATCH --job-name=med3_exp1_filtered
#SBATCH --partition=long
#SBATCH --output=logs/exp1_filtered_%j.log
#SBATCH --error=logs/exp1_filtered_error_%j.log
#SBATCH --nodes=1
# Tip: if you must target specific nodes, export SBATCH_NODELIST=gpuXYZ before calling sbatch.
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# =============================================================================
# SLURM Script: Experiment 1 with Lesion Size Filtering
# =============================================================================
# This script runs Experiment 1 with recommended lesion size filtering:
#   - min_voxels = 500 (cases with >= 500 voxels)
#   - min_dimension = 5 (minimum bounding box dimension >= 5)
#   - min_density = 0.3 (lesion density >= 30%)
#
# Methods trained:
#   - Med3-TabPFN (PFN-based classification head)
#   - Med3-LoCalPFN (Local context-aware PFN)
#   - DenseNet121-3D (3D baseline)
#   - ViT-3D (3D Vision Transformer baseline)
#
# Results are saved in separate folders from unfiltered experiments.
#
# Usage:
#   sbatch cluster_scripts/slurm/slurm_exp1_filtered.sh
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# -----------------------------------------------------------------------------
# 1. Path Configuration - UPDATE THESE FOR YOUR CLUSTER
# -----------------------------------------------------------------------------
# Path to your Med3Tab-PFN repository on the cluster
CODE_DIR="${SLURM_SUBMIT_DIR}"

# Path to your data directory (if different from code directory)
DATA_DIR="${CODE_DIR}"

# Where to save results
RESULTS_DIR="${CODE_DIR}/results/experiment1"

# Which config file to use
CONFIG_FILE="${CODE_DIR}/configs/datasets_cluster.yaml"

echo ""
echo "Code directory: $CODE_DIR"
echo "Data directory: $DATA_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Config file: $CONFIG_FILE"
echo ""
echo "⚠️  FILTERED MODE: Using recommended lesion size filtering"
echo "   min_voxels=500, min_dimension=5, min_density=0.3"
echo "   Results will be saved in separate '_filtered_*' folders per method/dataset"

# Create necessary directories
mkdir -p ${CODE_DIR}/logs
mkdir -p ${RESULTS_DIR}

# Change to code directory
cd ${CODE_DIR} || exit 1

# -----------------------------------------------------------------------------
# 2. Environment Setup
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 3. Run Experiment with Filtering
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Starting Experiment 1 with Filtering"
echo "=========================================="
echo ""
echo "Methods to run:"
echo "  1. Med3-TabPFN"
echo "  2. Med3-LoCalPFN"
echo "  3. DenseNet121-3D"
echo "  4. ViT-3D"
echo ""
echo "Filtering: ENABLED (recommended preset)"
echo ""

# Force unbuffered output for Python
export PYTHONUNBUFFERED=1

# Run with filtering enabled
python -u cluster_scripts/experiments/exp1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --epochs-3d ${EPOCHS:-20} \
    --filter-preset recommended

# Alternative: Use custom filtering parameters instead of preset
# Uncomment and modify these lines to use custom thresholds:
#    --min-voxels 500 \
#    --min-dimension 5 \
#    --min-density 0.3
#
# Enable ROI cropping (optional):
#    --use-roi-crop \
#    --roi-margin 10 \
#    --roi-target-size 128

# Capture exit status
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Experiment completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# -----------------------------------------------------------------------------
# 4. Results Summary
# -----------------------------------------------------------------------------
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: Filtered results saved to ${RESULTS_DIR}/"
    echo ""
    echo "Note: Results are in dataset-specific filtered folders:"
    echo "  e.g., tabpfn_runs/gist_filtered_v500_d5_ρ0.30/"
    echo "        localpfn_runs/gist_filtered_v500_d5_ρ0.30/"
    echo ""
    
    # List output files
    if [ -d "$RESULTS_DIR" ]; then
        echo "Output files:"
        ls -lh ${RESULTS_DIR}/ 2>/dev/null || echo "  (directory exists but empty)"
    fi
    
    # Display CSV summary if it exists
    SUMMARY_CSV="${RESULTS_DIR}/combined_benchmarks_summary.csv"
    AVG_CSV="${RESULTS_DIR}/average_scores_per_method.csv"
    
    if [ -f "$SUMMARY_CSV" ]; then
        echo ""
        echo "=========================================="
        echo "Detailed Results (All Runs):"
        echo "=========================================="
        cat $SUMMARY_CSV
        echo ""
    fi
    
    # Display average scores per method
    if [ -f "$AVG_CSV" ]; then
        echo ""
        echo "=========================================="
        echo "⭐ AVERAGE SCORES PER METHOD (FILTERED):"
        echo "=========================================="
        cat $AVG_CSV
        echo ""
        echo "(Sorted by ROC AUC - higher is better)"
        echo ""
    fi
    
    echo ""
    echo "To view results:"
    echo "  Detailed:  cat ${SUMMARY_CSV}"
    echo "  Averages:  cat ${AVG_CSV}"
    echo ""
    
else
    echo ""
    echo "❌ FAILED: Check error log at:"
    echo "  ${CODE_DIR}/logs/exp1_filtered_error_${SLURM_JOB_ID}.log"
    echo ""
fi

exit $EXIT_CODE
