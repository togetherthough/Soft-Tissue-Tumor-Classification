# Installation Guide for GeoTopo-STS

Complete installation instructions for all platforms and configurations.

---

## Prerequisites

- **Python**: 3.8 or higher (3.10 recommended)
- **CUDA**: 11.7+ (for GPU support)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: 8GB+ VRAM recommended for training

---

## Quick Install (Recommended)

### 1. Create Conda Environment

```bash
conda create -n geotopo python=3.10
conda activate geotopo
```

### 2. Install PyTorch

**For CUDA 11.8:**
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

**For other CUDA versions**, visit: https://pytorch.org/get-started/locally/

### 3. Install PyTorch Geometric

```bash
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

Adjust URL for your PyTorch + CUDA version.

### 4. Install GeoTopo-STS

```bash
cd geotopo_sts
pip install -e .
```

This installs the package in editable mode with all core dependencies.

### 5. Install Optional Dependencies

```bash
# For topology (choose one)
pip install gudhi  # Recommended
# OR
pip install giotto-tda

# For advanced features
pip install SimpleITK geoopt e3nn umap-learn wandb

# For development
pip install -e ".[dev]"
```

---

## Detailed Installation

### Option A: From Source (Full Control)

```bash
# Clone repository
git clone https://github.com/yourusername/geotopo-sts.git
cd geotopo-sts

# Create environment
conda create -n geotopo python=3.10
conda activate geotopo

# Install PyTorch (adjust for your system)
conda install pytorch=2.0.1 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse

# Install core dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Option B: Minimal Install (Core Only)

If you want to skip optional dependencies:

```bash
pip install torch numpy scipy scikit-learn scikit-image pyyaml
pip install nibabel trimesh torch-geometric
pip install tqdm pandas matplotlib seaborn

# Then install package
pip install -e . --no-deps
```

### Option C: Full Install (All Features)

```bash
pip install -e ".[full]"
```

This installs all optional dependencies including:
- SimpleITK (N4 bias correction)
- GUDHI (persistent homology)
- Geoopt (hyperbolic embeddings)
- E3NN (SE(3)-equivariant networks)
- PyMeshLab (advanced mesh operations)
- Weights & Biases (experiment tracking)
- UMAP (visualization)

---

## Platform-Specific Notes

### Linux (Ubuntu 20.04+)

Most straightforward installation. All dependencies work out of the box.

```bash
# If you need system dependencies for SimpleITK
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
```

### macOS

```bash
# Use conda for better compatibility
conda install -c conda-forge gudhi
conda install pytorch::pytorch torchvision -c pytorch

# Note: No CUDA support on macOS
# Model will run on CPU or MPS (Metal)
```

For M1/M2 Macs:
```bash
# Use native arm64 Python
conda create -n geotopo python=3.10
conda activate geotopo
pip install torch torchvision
```

### Windows

```bash
# Use Anaconda Prompt or PowerShell
conda create -n geotopo python=3.10
conda activate geotopo

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

**Note**: Some packages may require Visual Studio Build Tools.

---

## Troubleshooting

### 1. PyTorch Geometric Installation Fails

**Issue**: `torch-scatter` or `torch-sparse` won't install.

**Solution**:
```bash
# Install from pre-built wheels
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Or build from source (slower)
pip install torch-scatter torch-sparse --no-binary :all:
```

### 2. GUDHI Installation Fails

**Issue**: `gudhi` requires C++ compiler.

**Solutions**:

**Linux**:
```bash
sudo apt-get install build-essential cmake libboost-all-dev
pip install gudhi
```

**macOS**:
```bash
brew install cmake boost
pip install gudhi
```

**Windows**:
```bash
# Install Visual Studio Build Tools first
# Then try:
conda install -c conda-forge gudhi
```

**Alternative**: Use Giotto-TDA instead:
```bash
pip install giotto-tda
```

### 3. SimpleITK Issues

**Issue**: SimpleITK import errors.

**Solution**:
```bash
pip uninstall SimpleITK
pip install SimpleITK==2.2.1
```

Or skip it (bias correction will use approximation):
```python
# In config.yaml, set:
preprocessing:
  intensity:
    bias_correction: false
```

### 4. Out of Memory (OOM)

**Issue**: GPU runs out of memory during training.

**Solutions**:
- Reduce `batch_size` to 1
- Increase `gradient_accumulation` to 8 or 16
- Reduce `crop_size` to [128, 128, 128]
- Reduce mesh `target_vertices` to 5000
- Enable `mixed_precision: true`

### 5. Import Errors

**Issue**: `ModuleNotFoundError` after installation.

**Solution**:
```bash
# Ensure package is installed
pip install -e .

# Check installation
python -c "import geotopo_sts; print(geotopo_sts.__version__)"

# Re-install if needed
pip uninstall geotopo-sts
pip install -e .
```

### 6. CUDA Version Mismatch

**Issue**: `RuntimeError: CUDA error: no kernel image available`

**Solution**:
```bash
# Check your CUDA version
nvidia-smi

# Reinstall PyTorch with matching CUDA version
# For CUDA 11.7:
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117

# For CUDA 12.1:
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu121
```

---

## Verification

After installation, verify everything works:

```python
python -c "
import torch
import torch_geometric
import geotopo_sts

print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ PyTorch Geometric: {torch_geometric.__version__}')
print(f'✓ GeoTopo-STS: {geotopo_sts.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

Expected output:
```
✓ PyTorch: 2.0.1
✓ PyTorch Geometric: 2.3.1
✓ GeoTopo-STS: 0.1.0
✓ CUDA available: True
✓ CUDA device: NVIDIA GeForce RTX 3090
```

---

## Quick Test

Run a quick forward pass to ensure everything is working:

```bash
python -c "from geotopo_sts.example_usage import example_model_forward; example_model_forward()"
```

Or open the Jupyter notebook:
```bash
jupyter notebook QuickStart.ipynb
```

---

## Uninstallation

```bash
pip uninstall geotopo-sts
conda remove --name geotopo --all  # If you want to remove entire environment
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check GitHub Issues**: [github.com/yourusername/geotopo-sts/issues](https://github.com/yourusername/geotopo-sts/issues)
2. **Read the FAQ**: See `README.md` troubleshooting section
3. **Open a new issue**: Provide:
   - Your OS and Python version
   - Full error message
   - Output of `pip list`
   - CUDA version (if using GPU)

---

## Development Installation

For contributing to GeoTopo-STS:

```bash
# Clone with dev branch
git clone -b develop https://github.com/yourusername/geotopo-sts.git
cd geotopo-sts

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black geotopo_sts/
flake8 geotopo_sts/
```

---

## Docker Installation (Advanced)

A Dockerfile is provided for containerized deployment:

```bash
# Build image
docker build -t geotopo-sts .

# Run container
docker run --gpus all -v /path/to/data:/data geotopo-sts \
    geotopo-train --config /data/config.yaml --data /data/preprocessed
```

*(Docker support coming soon)*

---

**Installation complete!** 🎉 Proceed to the [Quick Start Guide](README.md#quick-start).
