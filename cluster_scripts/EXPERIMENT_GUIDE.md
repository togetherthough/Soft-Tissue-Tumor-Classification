# Running Experiment 1 on GPU Cluster

This guide explains how to run the benchmark experiment from `notebooks/Experiment1-Benchmarks.ipynb` on the BIGR GPU cluster using SLURM.

## Files Created

1. **`run_experiment1_benchmarks.py`** - Standalone Python script that runs the full experiment
2. **`slurm_experiment1_benchmarks.sh`** - SLURM batch script for job submission
3. **`CLUSTER_EXPERIMENT_GUIDE.md`** - This guide

## Prerequisites

### On the Cluster

1. **Code copied to**: `/trinity/home/r112276/Med3Tab-PFN`
2. **Data directory**: `/data/scratch/r112276`

### Data Organization

Ensure your data is organized under `/data/scratch/r112276/` as follows:

```
/data/scratch/r112276/
├── data/
│   ├── gist/
│   │   ├── sheet.csv
│   │   └── [your GIST data files]
│   └── lipo/
│       ├── sheet.csv
│       └── [your LIPO data files]
```

Or adjust the paths in the SLURM script's config section accordingly.

### Python Environment

You need to set up a Python environment with the required packages. Choose one of these options:

#### Option A: Using Conda/Mamba (Recommended)

```bash
# On the cluster login node
cd /trinity/home/r112276/Med3Tab-PFN

# Load conda module (if available)
module load Miniconda3  # or Anaconda3

# Create environment
conda create -n sammed3d python=3.10
conda activate sammed3d

# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other requirements
pip install -r hieracascade/requirements.txt
pip install -r med3pipe/requirements.txt

# Install additional packages for PFN methods
pip install tabpfn  # if using TabPFN
# Add any other method-specific dependencies
```

#### Option B: Using Python Module + pip

```bash
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.8.0

pip install --user -r hieracascade/requirements.txt
pip install --user -r med3pipe/requirements.txt
```

#### Option C: Using virtualenv

```bash
module load Python/3.10.8-GCCcore-12.2.0

cd /trinity/home/r112276/Med3Tab-PFN
python -m venv venv
source venv/bin/activate

pip install -r hieracascade/requirements.txt
pip install -r med3pipe/requirements.txt
```

## Configuration

### 1. Update SLURM Script

Edit `slurm_experiment1_benchmarks.sh`:

```bash
cd /trinity/home/r112276/Med3Tab-PFN
nano slurm_experiment1_benchmarks.sh
```

**Key settings to adjust:**

- **Email notifications** (line 10):
  ```bash
  #SBATCH --mail-user=your.email@erasmusmc.nl
  ```

- **Partition and Time** (lines 5-6):
  ```bash
  #SBATCH --partition=long        # Options: express, short, long
  #SBATCH --time=2-00:00:00       # Adjust based on expected runtime
  ```

- **Resources** (lines 7-9):
  ```bash
  #SBATCH --cpus-per-task=8       # Number of CPU cores
  #SBATCH --gres=gpu:1            # Number of GPUs (minimum 1)
  #SBATCH --mem=64G               # RAM per node
  ```

- **Environment activation** (around line 68): Uncomment and adjust based on your setup:
  ```bash
  # For conda:
  source /path/to/conda/etc/profile.d/conda.sh
  conda activate sammed3d
  
  # OR for virtualenv:
  source ${CODE_DIR}/venv/bin/activate
  ```

- **Data paths** (around lines 103-140): The script automatically creates a cluster-specific config. Verify the paths match your data organization.

### 2. Verify Data Paths

Make sure your data directories exist:

```bash
ls -la /data/scratch/r112276/data/gist/
ls -la /data/scratch/r112276/data/lipo/
```

Each should contain:
- `sheet.csv` (labels file)
- CT/MR image files organized as expected by your dataset

## Running the Experiment

### Option 1: Submit Job to Queue (Recommended)

```bash
cd /trinity/home/r112276/Med3Tab-PFN

# Make script executable
chmod +x slurm_experiment1_benchmarks.sh

# Submit job
sbatch slurm_experiment1_benchmarks.sh
```

The system will respond with a job ID:
```
Submitted batch job 123456
```

### Option 2: Run Specific Datasets Only

To run only specific datasets (e.g., just `gist`), edit the script and modify line 164:

```bash
srun python ${CODE_DIR}/run_experiment1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${CODE_DIR}/results/experiment1 \
    --datasets gist \
    --epochs-3d 10
```

### Option 3: Skip Certain Methods

You can skip methods by adding flags:

```bash
srun python ${CODE_DIR}/run_experiment1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${CODE_DIR}/results/experiment1 \
    --skip-baselines \      # Skip DenseNet121 and ViT
    --epochs-3d 10
```

Available skip flags:
- `--skip-tabpfn` - Skip TabPFN method
- `--skip-localpfn` - Skip LoCalPFN method
- `--skip-baselines` - Skip 3D baseline methods (DenseNet121 & ViT)

### Option 4: Interactive Testing (for debugging)

For quick testing before submitting a batch job:

```bash
# Request an interactive GPU session
srun --partition=interactive --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash

# Then run the script manually
cd /trinity/home/r112276/Med3Tab-PFN
conda activate sammed3d  # or your environment

python run_experiment1_benchmarks.py \
    --config configs/datasets_cluster.yaml \
    --output-dir results/experiment1 \
    --datasets gist \
    --epochs-3d 2
```

## Monitoring Your Job

### Check Queue Status

```bash
# View all your jobs
squeue -u r112276

# View specific job
squeue -j 123456

# Detailed job info
scontrol show job 123456
```

### View Logs (while running)

```bash
# Output log
tail -f /trinity/home/r112276/Med3Tab-PFN/logs/experiment1_benchmarks_123456.log

# Error log
tail -f /trinity/home/r112276/Med3Tab-PFN/logs/experiment1_benchmarks_error_123456.log
```

### Cancel Job

```bash
scancel 123456
```

### Check Job Efficiency (after completion)

```bash
seff 123456
```

## Results

Results will be saved to: `/trinity/home/r112276/Med3Tab-PFN/results/experiment1/`

### Output Files

1. **`combined_benchmarks_summary.csv`** - Main results table with all methods
2. **`tabpfn_runs/`** - TabPFN detailed results per dataset
3. **`localpfn_runs/`** - LoCalPFN detailed results per dataset
4. **`baselines/`** - 3D baseline model results per dataset

### View Results

```bash
# View summary
cat /trinity/home/r112276/Med3Tab-PFN/results/experiment1/combined_benchmarks_summary.csv

# Or with column formatting
column -t -s, /trinity/home/r112276/Med3Tab-PFN/results/experiment1/combined_benchmarks_summary.csv
```

### Download Results to Local Machine

From your local machine:

```bash
# Download all results
scp -r r112276@gpu-login.erasmusmc.nl:/trinity/home/r112276/Med3Tab-PFN/results/experiment1 ./

# Or just the summary
scp r112276@gpu-login.erasmusmc.nl:/trinity/home/r112276/Med3Tab-PFN/results/experiment1/combined_benchmarks_summary.csv ./
```

## Troubleshooting

### Job Fails Immediately

1. Check error log:
   ```bash
   cat /trinity/home/r112276/Med3Tab-PFN/logs/experiment1_benchmarks_error_*.log
   ```

2. Common issues:
   - **Module not found**: Check environment activation in SLURM script
   - **Data not found**: Verify data paths in config
   - **GPU not available**: Check `#SBATCH --gres=gpu:1` is set
   - **Out of memory**: Increase `#SBATCH --mem=` value

### Job Pending for Long Time

- Check queue status: `squeue -u r112276`
- Common reasons (shown in `NODELIST(REASON)`):
  - `Priority`: Other jobs have higher priority, wait
  - `Resources`: Waiting for resources, wait
  - `QOSMaxGRESPerUser`: You've reached GPU limit, wait or cancel other jobs

### CUDA Out of Memory

Reduce batch sizes or model sizes in the code, or request more GPU memory:

```bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_mem_24gb  # Request specific GPU with more memory
```

### Python Import Errors

Ensure all packages are installed:

```bash
# Activate your environment
conda activate sammed3d

# Verify key packages
python -c "import torch; import med3pipe; print('OK')"
```

### SAM-Med3D Checkpoint Missing

The script expects SAM-Med3D checkpoints. Ensure you have:

```bash
ls /trinity/home/r112276/Med3Tab-PFN/SAM-Med3D-main/SAM-Med3D-main/work_dir/
```

If missing, you may need to download or adjust paths in your code.

## Advanced Usage

### Running Multiple Experiments in Parallel

You can submit multiple jobs with different configurations:

```bash
# Job 1: TabPFN only
sbatch --export=ALL,SKIP_LOCALPFN=1,SKIP_BASELINES=1 slurm_experiment1_benchmarks.sh

# Job 2: LoCalPFN only  
sbatch --export=ALL,SKIP_TABPFN=1,SKIP_BASELINES=1 slurm_experiment1_benchmarks.sh

# Job 3: Baselines only
sbatch --export=ALL,SKIP_TABPFN=1,SKIP_LOCALPFN=1 slurm_experiment1_benchmarks.sh
```

(Note: You'll need to modify the script to read these environment variables)

### Array Jobs (for multiple datasets)

If you want to run each dataset as a separate job:

```bash
#SBATCH --array=0-1  # For 2 datasets

# Then in the script:
DATASETS=(gist lipo)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

srun python run_experiment1_benchmarks.py \
    --datasets $DATASET \
    ...
```

## Resource Recommendations

Based on the experiment type:

### Quick Test (2 datasets, low epochs)
```bash
#SBATCH --partition=short
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
```

### Full Benchmark (all datasets, 10+ epochs)
```bash
#SBATCH --partition=long
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
```

### Large-Scale (many datasets, high resolution)
```bash
#SBATCH --partition=long
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
```

## Contact

For cluster-specific issues, contact BIGR cluster support.
For code/experiment issues, check the main repository documentation.

---

**Last Updated**: $(date)
