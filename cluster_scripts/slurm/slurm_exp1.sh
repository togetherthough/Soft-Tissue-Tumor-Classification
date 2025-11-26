#!/bin/bash
#SBATCH --job-name=med3_exp1
#SBATCH --partition=long
#SBATCH --output=logs/exp1_%j.log
#SBATCH --error=logs/exp1_error_%j.log
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
# SLURM Script: Experiment 1 - Train & Test Med3-TabPFN Methods vs Baselines
# =============================================================================
# This script trains and evaluates:
#   - Med3-TabPFN (PFN-based classification head)
#   - Med3-LoCalPFN (Local context-aware PFN)
#   - DenseNet121-3D (3D baseline)
#   - ViT-3D (3D Vision Transformer baseline)
#
# Usage:
#   sbatch cluster_scripts/slurm/slurm_exp1.sh
#
# Configure the paths below for your cluster environment.
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
# If your data is in the same repo, you can use: DATA_DIR="${CODE_DIR}"
DATA_DIR="${CODE_DIR}"

# Where to save results
RESULTS_DIR="${CODE_DIR}/results/experiment1"

# Which config file to use
# Use cluster-specific config with absolute paths to /data/scratch/
CONFIG_FILE="${CODE_DIR}/configs/datasets_cluster.yaml"

echo ""
echo "Code directory: $CODE_DIR"
echo "Data directory: $DATA_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Config file: $CONFIG_FILE"

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
module load Python/3.9
module load CUDA/12.3.0

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

# -----------------------------------------------------------------------------
# 3. Run Experiment
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Starting Experiment 1: Train & Test"
echo "=========================================="
echo ""
echo "Methods to run:"
echo "  1. Med3-TabPFN"
echo "  2. Med3-LoCalPFN"
echo "  3. DenseNet121-3D"
echo "  4. ViT-3D"
echo ""

# Run the experiment
# Available options:
#   --datasets gist lipo        : run only specific datasets
#   --skip-tabpfn               : skip TabPFN method
#   --skip-localpfn             : skip LoCalPFN method
#   --skip-baselines            : skip 3D baselines (DenseNet & ViT)
#   --epochs-3d N               : number of training epochs for 3D models

# Force unbuffered output for Python
export PYTHONUNBUFFERED=1

python -u cluster_scripts/experiments/exp1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --epochs-3d ${EPOCHS:-20}  # Can override with EPOCHS env var

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
    echo "✅ SUCCESS: Results saved to ${RESULTS_DIR}/"
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
        echo "⭐ AVERAGE SCORES PER METHOD:"
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
    echo "  ${CODE_DIR}/logs/exp1_error_${SLURM_JOB_ID}.log"
    echo ""
fi

exit $EXIT_CODE
