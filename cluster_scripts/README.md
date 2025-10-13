# Cluster Scripts for Experiment 1 Benchmarks

This directory contains all files needed to run **Experiment 1: Head-to-head Benchmarks** on the BIGR GPU cluster using SLURM.

## 📁 Files in This Directory

| File | Description |
|------|-------------|
| **`run_experiment1_benchmarks.py`** | Standalone Python script that runs the full experiment |
| **`slurm_experiment1.sh`** | SLURM batch script for job submission |
| **`check_setup.sh`** | Environment verification script |
| **`EXPERIMENT_GUIDE.md`** | Comprehensive documentation and setup guide |
| **`QUICK_START.md`** | Condensed quick start guide (5 steps) |
| **`COMMANDS.txt`** | Command reference card for common operations |
| **`README.md`** | This file |

## 🚀 Quick Start

### On the Cluster

```bash
# 1. Navigate to the code directory
cd /trinity/home/r112276/Med3Tab-PFN/cluster_scripts

# 2. Make scripts executable
chmod +x slurm_experiment1.sh check_setup.sh

# 3. Verify environment (optional but recommended)
bash check_setup.sh

# 4. Edit SLURM script - update email and environment activation
nano slurm_experiment1.sh

# 5. Submit job
sbatch slurm_experiment1.sh
```

## 📖 Documentation

- **New users**: Start with `QUICK_START.md` (5-step guide)
- **Detailed setup**: Read `EXPERIMENT_GUIDE.md` (comprehensive)
- **Command reference**: Check `COMMANDS.txt` (quick lookup)

## 🔧 What the Experiment Does

Compares 4 methods across your datasets:
1. **Med3-TabPFN** - TabPFN with Med3D embeddings
2. **Med3-LoCalPFN** - LoCalPFN with Med3D embeddings
3. **DenseNet121-3D** - 3D baseline
4. **ViT-3D** - 3D baseline

## 📊 Output Location

Results will be saved to:
```
/trinity/home/r112276/Med3Tab-PFN/results/experiment1/
├── combined_benchmarks_summary.csv  # Main results table
├── tabpfn_runs/                      # TabPFN details
├── localpfn_runs/                    # LoCalPFN details
└── baselines/                        # DenseNet & ViT outputs
```

## 🔍 Monitor Your Job

```bash
# Check status
squeue -u r112276

# View live output log (replace JOBID with your job ID)
tail -f ../logs/experiment1_benchmarks_JOBID.log

# View error log
tail -f ../logs/experiment1_benchmarks_error_JOBID.log
```

## ⚙️ Customization

Edit `slurm_experiment1.sh` to:
- Adjust resources (CPU, GPU, memory, time)
- Run specific datasets only
- Skip certain methods
- Change training epochs

See `EXPERIMENT_GUIDE.md` for details.

## ⚠️ Important Notes

1. **Update email** in `slurm_experiment1.sh` (line 11)
2. **Activate environment** - uncomment and fix lines 68-72 in `slurm_experiment1.sh`
3. **Verify data paths** - ensure `/data/scratch/r112276/data/{gist,lipo}/` exists
4. **GPU required** - at least 1 GPU must be requested

## 🆘 Troubleshooting

1. **Job fails immediately**: Check error log and verify environment activation
2. **Import errors**: Ensure all Python packages are installed
3. **CUDA unavailable**: Normal on login node; compute nodes have GPUs
4. **Out of memory**: Increase `--mem` in SLURM script

For detailed troubleshooting, see `EXPERIMENT_GUIDE.md`.

## 📞 Getting Help

- **Setup issues**: Run `bash check_setup.sh`
- **Command help**: See `COMMANDS.txt`
- **Cluster issues**: Contact BIGR cluster support
- **Code issues**: Check main repository documentation

---

**Ready to run?**

```bash
cd /trinity/home/r112276/Med3Tab-PFN/cluster_scripts
sbatch slurm_experiment1.sh
```
