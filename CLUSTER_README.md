# GPU Cluster Scripts

All scripts and documentation for running experiments on the GPU cluster are located in the **`cluster_scripts/`** directory.

## 📂 Location

```
Med3Tab-PFN/
└── cluster_scripts/
    ├── README.md                          # Start here
    ├── QUICK_START.md                     # 5-step quick guide
    ├── EXPERIMENT_GUIDE.md                # Comprehensive documentation
    ├── COMMANDS.txt                       # Command reference
    ├── run_experiment1_benchmarks.py      # Python experiment script
    ├── slurm_experiment1.sh               # SLURM batch script
    └── check_setup.sh                     # Environment verification
```

## 🚀 Quick Start

```bash
cd cluster_scripts/
cat README.md
```

Or on the cluster:

```bash
cd /trinity/home/r112276/Med3Tab-PFN/cluster_scripts
bash check_setup.sh
sbatch slurm_experiment1.sh
```

## 📖 Documentation

1. **New users**: Read `cluster_scripts/QUICK_START.md`
2. **Detailed guide**: See `cluster_scripts/EXPERIMENT_GUIDE.md`
3. **Command reference**: Check `cluster_scripts/COMMANDS.txt`

---

**All cluster-related files are now organized in the `cluster_scripts/` folder.**
