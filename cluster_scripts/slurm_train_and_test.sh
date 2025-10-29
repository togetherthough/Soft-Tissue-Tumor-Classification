#!/bin/bash
#SBATCH --job-name=med3_exp1
#SBATCH --output=logs/exp1_%j.log
#SBATCH --error=logs/exp1_error_%j.log
#SBATCH --partition=long
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
#   sbatch cluster_scripts/slurm_train_and_test.sh
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

# Option A: Load modules (uncomment and adjust for your cluster)
# module purge
# module load Python/3.10.8
# module load CUDA/11.8.0
# module load cuDNN/8.7.0.84-CUDA-11.8.0

# Option B: Activate conda environment (recommended)
# Adjust the path to your conda installation
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate sammed3d

# Option C: Activate virtualenv
# source ${CODE_DIR}/venv/bin/activate

# Set threading environment variables for optimal performance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Optional: Limit to first GPU if multiple are available
export CUDA_VISIBLE_DEVICES=0

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

python cluster_scripts/run_experiment1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --epochs-3d 10

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
    if [ -f "$SUMMARY_CSV" ]; then
        echo ""
        echo "=========================================="
        echo "Results Summary:"
        echo "=========================================="
        cat $SUMMARY_CSV
        echo ""
    fi
    
    echo ""
    echo "To view detailed results:"
    echo "  cat ${SUMMARY_CSV}"
    echo ""
    
else
    echo ""
    echo "❌ FAILED: Check error log at:"
    echo "  ${CODE_DIR}/logs/exp1_error_${SLURM_JOB_ID}.log"
    echo ""
fi

exit $EXIT_CODE
