#!/bin/bash
# Run all three-way comparison experiments on Linux/cluster
# Compares: Baseline, Filtered Baseline, and ROI-Cropped

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "======================================================================"
echo "THREE-WAY COMPARISON EXPERIMENTS"
echo "======================================================================"
echo "Comparing:"
echo "  1. Baseline (full-volume, no filtering)"
echo "  2. Filtered Baseline (full-volume + lesion filtering)"
echo "  3. ROI-Cropped (adaptive crop/pad)"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️  Warning: nvidia-smi not found. Running on CPU."
    echo ""
fi

# Experiment 1: TabPFN Three-Way
echo "======================================================================"
echo "EXPERIMENT 1: TabPFN (Three-Way Comparison)"
echo "======================================================================"
echo ""

python "$SCRIPT_DIR/run_experiment_tabpfn_roi_comparison.py"

echo ""
echo "✅ TabPFN three-way comparison complete"
echo ""

# Experiment 2: LoCalPFN Three-Way
echo "======================================================================"
echo "EXPERIMENT 2: LoCalPFN (Three-Way Comparison)"
echo "======================================================================"
echo ""

python "$SCRIPT_DIR/run_experiment_localpfn_three_way.py"

echo ""
echo "✅ LoCalPFN three-way comparison complete"
echo ""

# Summary
echo "======================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  - TabPFN:    $PROJECT_ROOT/results/three_way_comparison/"
echo "  - LoCalPFN:  $PROJECT_ROOT/results/three_way_comparison_localpfn/"
echo ""
echo "To view results:"
echo "  cat results/three_way_comparison/three_way_comparison.csv"
echo "  cat results/three_way_comparison_localpfn/three_way_comparison.csv"
echo ""
