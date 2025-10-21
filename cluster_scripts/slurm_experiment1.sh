#!/bin/bash
#SBATCH --job-name=exp1_benchmarks
#SBATCH --output=/trinity/home/r112276/Med3Tab-PFN/logs/experiment1_benchmarks_%j.log
#SBATCH --error=/trinity/home/r112276/Med3Tab-PFN/logs/experiment1_benchmarks_error_%j.log
#SBATCH --partition=long
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# =============================================================================
# SLURM Job Script for Experiment 1: Head-to-head Benchmarks
# =============================================================================
# This script runs the benchmark experiment comparing:
#   - Med3-TabPFN
#   - Med3-LoCalPFN
#   - DenseNet121-3D
#   - ViT-3D
#
# Adjust the SBATCH parameters above according to your needs:
#   - partition: short/long/express depending on expected runtime
#   - time: total wall time (format: days-hours:minutes:seconds)
#   - cpus-per-task: number of CPU cores
#   - mem: RAM per node
#   - gres=gpu:X: number of GPUs (at least 1 required)
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# -----------------------------------------------------------------------------
# Directory Setup
# -----------------------------------------------------------------------------
CODE_DIR="/trinity/home/r112276/Med3Tab-PFN"
DATA_DIR="/data/scratch/r112276"
SCRIPT_DIR="${CODE_DIR}/cluster_scripts"

echo ""
echo "Code directory: $CODE_DIR"
echo "Data directory: $DATA_DIR"
echo "Script directory: $SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p ${CODE_DIR}/logs
mkdir -p ${CODE_DIR}/results/experiment1

# Change to code directory
cd ${CODE_DIR} || exit 1

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Setting up environment..."
echo "=========================================="

# Load required modules (adjust based on your cluster's available modules)
# Uncomment and modify as needed:
# module purge
# module load Python/3.10.8-GCCcore-12.2.0
# module load CUDA/11.8.0
# module load cuDNN/8.7.0.84-CUDA-11.8.0

# Activate Python environment
# Option 1: If using conda/mamba
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate sammed3d

# Option 2: If using virtualenv
# source ${CODE_DIR}/venv/bin/activate

# Option 3: If using module-based Python, ensure packages are installed
# pip install --user -r ${CODE_DIR}/med3pipe/requirements.txt

# Verify Python and CUDA
echo ""
echo "Python version:"
python --version

echo ""
echo "PyTorch version and CUDA availability:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Set environment variables for optimal performance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# CUDA settings (optional, adjust as needed)
export CUDA_VISIBLE_DEVICES=0

# -----------------------------------------------------------------------------
# Update Config to Point to Cluster Data
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Configuring dataset paths..."
echo "=========================================="

# Create a cluster-specific config that points to the data directory
# You may need to adjust this based on your actual data organization
CONFIG_FILE="${CODE_DIR}/configs/datasets_cluster.yaml"

cat > ${CONFIG_FILE} << EOF
datasets:
  gist:
    dataset_root: ${DATA_DIR}/data/gist
    category: gist
    ct_name: ct_GIST
    labels:
      sheet_csv: sheet.csv
      dataset_name: GIST
      subject_col: Subject
      label_col: Diagnosis_binary
      case_suffix: _CT
    prepare:
      case_glob: null
      max_cases: null
    split:
      ratio: 0.8
      seed: 2025
    extraction:
      img_size: 128

  lipo:
    dataset_root: ${DATA_DIR}/data/lipo
    category: lipo
    ct_name: ct_LIPO
    labels:
      sheet_csv: sheet.csv
      dataset_name: null
      subject_col: Subject
      label_col: Diagnosis_binary
      case_suffix: _MR
    prepare:
      case_glob: null
      max_cases: null
    split:
      ratio: 0.8
      seed: 2025
    extraction:
      img_size: 128
EOF

echo "Created cluster config: ${CONFIG_FILE}"

# Alternative: If you want to use the original config but have data symlinked,
# you can just use the original config file:
# CONFIG_FILE="${CODE_DIR}/configs/datasets.yaml"

# -----------------------------------------------------------------------------
# Run Experiment
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Starting Experiment 1: Benchmarks"
echo "=========================================="
echo ""

# Run the experiment script
# Adjust arguments as needed:
#   --datasets gist lipo      : run only specific datasets
#   --skip-tabpfn             : skip TabPFN method
#   --skip-localpfn           : skip LoCalPFN method
#   --skip-baselines          : skip 3D baselines
#   --epochs-3d 10            : use more epochs for 3D models

srun python ${SCRIPT_DIR}/run_experiment1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${CODE_DIR}/results/experiment1 \
    --epochs-3d 10

# Capture exit status
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Experiment completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# -----------------------------------------------------------------------------
# Cleanup and Summary
# -----------------------------------------------------------------------------
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Results saved to ${CODE_DIR}/results/experiment1/"
    echo ""
    echo "Output files:"
    ls -lh ${CODE_DIR}/results/experiment1/
    
    # Display summary if CSV exists
    SUMMARY_CSV="${CODE_DIR}/results/experiment1/combined_benchmarks_summary.csv"
    if [ -f "$SUMMARY_CSV" ]; then
        echo ""
        echo "Summary Results:"
        cat $SUMMARY_CSV
    fi
else
    echo ""
    echo "FAILED: Check error log at:"
    echo "  /trinity/home/r112276/Med3Tab-PFN/logs/experiment1_benchmarks_error_${SLURM_JOB_ID}.log"
fi

exit $EXIT_CODE
