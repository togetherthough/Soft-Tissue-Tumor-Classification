# Documentation Map

Complete navigation guide to all project documentation - find what you need quickly by task, keyword, or topic.

## Quick Navigation by Task

### I want to...

#### Run Experiments
- **Run Experiment 3 (Classification Head)** → [`docs/experiments/experiment3/README.md`](docs/experiments/experiment3/README.md)
  - All running options (3/5/10/20 epochs)
  - Local vs cluster instructions
  - Configuration parameters
  - Expected results by epoch count

#### Troubleshoot Errors
- **Fix Experiment 3 errors** → [`docs/experiments/experiment3/troubleshooting.md`](docs/experiments/experiment3/troubleshooting.md)
  - "invalid load key" (corrupted checkpoint)
  - "No rows found" (dataset name mismatch)
  - CUDA out of memory
  - Config duplicates
- **General issues** → [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)

#### Understand Technical Details
- **Embedding extraction/reuse** → [`docs/technical/embeddings.md`](docs/technical/embeddings.md)
  - `skip_existing_embeddings` parameter explained
  - Per-dataset automatic extraction
  - Time savings guide
- **Preprocessing pipeline** → [`docs/technical/preprocessing.md`](docs/technical/preprocessing.md)
  - ResizeLargestTo + CropOrPad logic
  - Why this approach is correct
- **SAM Dice score evaluation** → [`docs/technical/sam-dice-scores.md`](docs/technical/sam-dice-scores.md)
  - 11-click iterative refinement
  - Debugging low scores
  - Prompt strategy recommendations

#### Setup Environment
- **Cluster setup (HPC)** → [`docs/setup/cluster-setup.md`](docs/setup/cluster-setup.md)
  - Virtual environment activation
  - SLURM scripts overview
  - Exit code 127 fixes

#### Analyze Results
- **Visualization issues** → [`docs/analysis/visualization-notes.md`](docs/analysis/visualization-notes.md)
  - Sagittal padding explanation
  - Voxel spacing and physical volumes
- **SAM feature quality** → [`docs/analysis/sam-feature-evaluation.md`](docs/analysis/sam-feature-evaluation.md)
  - Feature discrimination analysis

#### Quick Reference
- **Common commands** → [`docs/QUICK_REFERENCE.md`](docs/QUICK_REFERENCE.md)
- **Change history** → [`docs/CHANGES_SUMMARY.md`](docs/CHANGES_SUMMARY.md)

## Documentation Structure

```
docs/
├── experiments/          # 🧪 How to run experiments
│   └── experiment3/
│       ├── README.md              # Complete experiment guide
│       └── troubleshooting.md     # Error fixes & debugging
│
├── technical/           # 🔧 Technical implementation
│   ├── embeddings.md              # Embedding extraction & reuse
│   ├── preprocessing.md           # Data preprocessing pipeline
│   ├── multi-dataset.md           # Multi-dataset workflows
│   ├── sam-dice-scores.md         # SAM evaluation & debugging
│   └── pipeline-testing.md        # End-to-end pipeline testing
│
├── setup/              # ⚙️ Environment configuration
│   └── cluster-setup.md           # Cluster/HPC setup
│
├── analysis/           # 📊 Data analysis & visualization
│   ├── visualization-notes.md     # Spacing, padding, volumes
│   └── sam-feature-evaluation.md  # SAM feature quality analysis
│
├── QUICK_REFERENCE.md   # ⚡ Quick commands cheat sheet
├── TROUBLESHOOTING.md   # 🔍 General troubleshooting
├── CHANGES_SUMMARY.md   # 📝 Major project changes
└── README.md           # 📖 Main documentation index
```

## Common Scenarios

### "I'm getting an error when running Experiment 3"
1. Check [`docs/experiments/experiment3/troubleshooting.md`](docs/experiments/experiment3/troubleshooting.md)
2. Look for your specific error message
3. Follow the step-by-step fix instructions
4. Run the diagnostic script: `bash cluster_scripts/diagnose_experiment3.sh`

### "I want to save time by reusing embeddings"
1. Read [`docs/technical/embeddings.md`](docs/technical/embeddings.md)
2. Use `skip_existing_embeddings=False` (default)
3. Pipeline automatically detects missing embeddings per dataset
4. Can save ~20+ minutes per run

### "My sagittal views look all gray/padded"
1. See [`docs/analysis/visualization-notes.md`](docs/analysis/visualization-notes.md) → Sagittal Padding section
2. Understand Z-axis padding after preprocessing
3. Use auto-detect function to find data boundaries
4. Adjust visualization indices to data region

### "I need to run experiments on the cluster"
1. Read [`docs/setup/cluster-setup.md`](docs/setup/cluster-setup.md)
2. Verify environment activation commands
3. Check SLURM script configuration
4. Submit with `sbatch cluster_scripts/slurm_experiment3_quick.sh`

### "SAM Dice scores are very low"
1. Check [`docs/technical/sam-dice-scores.md`](docs/technical/sam-dice-scores.md) → Debugging section
2. Review root cause analysis (spatial misalignment)
3. Consider implementing multi-point initialization
4. Run debug mode: `--debug-case dataset:case_id`

## Detailed Content Guide

### 📁 `docs/experiments/experiment3/`

#### `README.md` - Comprehensive Experiment 3 Guide
- **Quick start** (local & cluster)
- **Running with different epochs** (5 methods)
- **Configuration parameters** (all CLI arguments)
- **Expected results** by epoch count (3/5/10/20)
- **Recommended testing strategy** (3 phases)
- **Diagnostic tools** and monitoring
- **Output structure** and success indicators

#### `troubleshooting.md` - Complete Error Reference
- **Common errors** with solutions:
  - "invalid load key, 'E'." → Checkpoint corruption fix
  - "No rows found" → Dataset name matching
  - CUDA out of memory → Batch size reduction
  - Config duplicates → Validation and cleanup
- **Pre-flight checklist** (7 items)
- **Diagnostic commands** (environment, files, logs)
- **Step-by-step debugging** sequence (5 steps)
- **Performance optimization** (speed & accuracy)
- **FAQ** with quick answers

### 📁 `docs/technical/`

#### `embeddings.md` - Embedding Extraction & Reuse
- **Parameter explanation** (`skip_existing_embeddings`)
- **When to use each setting** (False vs True)
- **Automatic per-dataset extraction** logic
- **Benefits** (order-independent, smart caching)
- **Time savings** (20+ min per run)
- **Bias analysis** (why reusing is safe)
- **Troubleshooting** common issues

#### `preprocessing.md` - Data Preprocessing Pipeline
- **ResizeLargestTo + CropOrPad** approach
- **Why this is correct** (preserves all data)
- **Step-by-step transformation** examples
- **Comparison with alternatives**
- **Modality-aware processing** (CT vs MR)

#### `sam-dice-scores.md` - SAM Evaluation
- **11-click iterative refinement** protocol
- **Output format** and metrics
- **Debugging low Dice scores**:
  - Root cause analysis (spatial misalignment)
  - Per-click metrics analysis
  - Contributing factors (downsampling, prompts)
- **Recommendations** (prioritized):
  - Multi-point initialization
  - Bounding box prompts
  - Adaptive click sampling
- **Tools** for fast testing and debugging

#### `multi-dataset.md` - Multi-Dataset Workflows
- Running experiments across datasets
- Configuration management
- Result aggregation

#### `pipeline-testing.md` - End-to-End Testing
- Quick test (10 cases, ~5-10 min)
- Full pipeline validation
- Verification points (8 checks)
- Output structure

### 📁 `docs/setup/`

#### `cluster-setup.md` - HPC Environment
- **Virtual environment activation** (module + venv)
- **SLURM scripts overview** (5 scripts)
- **Common issues**:
  - Exit code 127 (command not found)
  - Wrong environment name
- **Submitting jobs** and monitoring
- **Expected output** on success
- **Troubleshooting** failed jobs
- **Local vs cluster** config differences

### 📁 `docs/analysis/`

#### `visualization-notes.md` - Visualization & Measurements
- **Sagittal padding explanation**:
  - Why views look gray (Z-axis padding)
  - Verification at different indices
  - 3 solution options (quick fix, auto-detect, window adjust)
- **Voxel spacing**:
  - Importance for accurate volumes
  - SimpleITK conventions
  - Physical dimensions calculation
  - Updated notebooks (2 files)

#### `sam-feature-evaluation.md` - SAM Feature Analysis
- Feature discrimination quality
- Baseline performance metrics
- Decision criteria (AUC thresholds)

### 📁 Root-Level Docs

#### `QUICK_REFERENCE.md` - Command Cheat Sheet
- Most common commands
- Quick copy-paste reference

#### `TROUBLESHOOTING.md` - General Issues
- Issues not specific to Experiment 3
- General pipeline problems
- Environment setup issues

#### `CHANGES_SUMMARY.md` - Project History
- Major changes and updates
- Breaking changes
- Migration notes

## Files by Topic

### Getting Started
- **Experiment 3**: `docs/experiments/experiment3/README.md` - How to run, all options
- **Cluster Setup**: `docs/setup/cluster-setup.md` - Environment configuration
- **Quick Reference**: `docs/QUICK_REFERENCE.md` - Common commands

### Technical Details
- **Embeddings**: `docs/technical/embeddings.md` - Extraction, reuse, per-dataset logic
- **Preprocessing**: `docs/technical/preprocessing.md` - ResizeLargestTo + CropOrPad pipeline
- **Multi-Dataset**: `docs/technical/multi-dataset.md` - Running multiple datasets
- **SAM Dice Scores**: `docs/technical/sam-dice-scores.md` - SAM evaluation
- **Pipeline Testing**: `docs/technical/pipeline-testing.md` - End-to-end testing

### Troubleshooting
- **Experiment 3 Issues**: `docs/experiments/experiment3/troubleshooting.md` - All error fixes
- **General Issues**: `docs/TROUBLESHOOTING.md` - General troubleshooting

### Analysis & Visualization
- **Visualization Notes**: `docs/analysis/visualization-notes.md` - Spacing, padding, volumes
- **SAM Features**: `docs/analysis/sam-feature-evaluation.md` - Feature quality analysis

## Search by Keyword

### Errors & Debugging
- **"invalid load key"** → `docs/experiments/experiment3/troubleshooting.md`
- **"No rows found"** → `docs/experiments/experiment3/troubleshooting.md`
- **"CUDA out of memory"** → `docs/experiments/experiment3/troubleshooting.md`
- **Low Dice scores** → `docs/technical/sam-dice-scores.md`
- **Exit code 127** → `docs/setup/cluster-setup.md`

### Configuration
- **Epochs (3/5/10/20)** → `docs/experiments/experiment3/README.md`
- **Batch size** → `docs/experiments/experiment3/README.md`
- **Learning rate** → `docs/experiments/experiment3/README.md`
- **Dataset config** → `docs/technical/multi-dataset.md`
- **SLURM scripts** → `docs/setup/cluster-setup.md`

### Performance & Optimization
- **Embedding reuse** → `docs/technical/embeddings.md`
- **Time savings** → `docs/technical/embeddings.md`
- **Frozen vs fine-tune** → `docs/experiments/experiment3/README.md`
- **Speed optimization** → `docs/experiments/experiment3/troubleshooting.md`

### Data & Preprocessing
- **ResizeLargestTo** → `docs/technical/preprocessing.md`
- **CropOrPad** → `docs/technical/preprocessing.md`
- **Voxel spacing** → `docs/analysis/visualization-notes.md`
- **Physical volumes** → `docs/analysis/visualization-notes.md`
- **Padding explanation** → `docs/analysis/visualization-notes.md`

### Evaluation
- **SAM segmentation** → `docs/technical/sam-dice-scores.md`
- **Iterative refinement** → `docs/technical/sam-dice-scores.md`
- **Feature quality** → `docs/analysis/sam-feature-evaluation.md`
- **AUC interpretation** → `docs/analysis/sam-feature-evaluation.md`

## Additional Resources

### Scripts Directory
- **Cluster scripts**: `cluster_scripts/` - SLURM batch scripts and Python runners
  - `EXPERIMENT3_GUIDE.md` - Original experiment 3 guide
  - `README.md` - Cluster scripts overview
  - `diagnose_experiment3.sh` - Diagnostic tool

### Notebooks Directory
- **Visualization notebooks**: `notebooks/visualization/`
  - `SAM_Visualization.ipynb` - SAM segmentation analysis
  - `Dataset_Analysis.ipynb` - Dataset statistics with physical volumes
  - `Raw_Image_Display.ipynb` - Raw data inspection

### Config Files
- **Dataset configs**: `configs/`
  - `datasets.yaml` - Local configuration
  - `datasets_cluster.yaml` - Cluster configuration
  - `datasets_analysis.yaml` - Analysis configuration

## Quick Start Paths

### New User
1. Start: [`docs/README.md`](docs/README.md) - Documentation index
2. Then: [`docs/experiments/experiment3/README.md`](docs/experiments/experiment3/README.md) - Run first experiment
3. If errors: [`docs/experiments/experiment3/troubleshooting.md`](docs/experiments/experiment3/troubleshooting.md)

### Cluster User
1. Start: [`docs/setup/cluster-setup.md`](docs/setup/cluster-setup.md) - Environment setup
2. Then: [`docs/experiments/experiment3/README.md`](docs/experiments/experiment3/README.md) - Submit jobs
3. Monitor: Check SLURM logs and use diagnostic script

### Developer
1. Technical: [`docs/technical/`](docs/technical/) - All implementation details
2. Testing: [`docs/technical/pipeline-testing.md`](docs/technical/pipeline-testing.md) - Test your changes
3. Analysis: [`docs/analysis/`](docs/analysis/) - Understand results

---

**Last Updated**: November 24, 2025  
**Total Docs**: 12 comprehensive guides organized in 4 categories  
**Navigation**: Start at [`docs/README.md`](docs/README.md) for full index
