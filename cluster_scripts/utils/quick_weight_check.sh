#!/bin/bash
# Quick script to verify SAM-Med3D weights on the cluster

echo "=========================================="
echo "SAM-Med3D Weight Verification"
echo "=========================================="

CKPT_PATH="/trinity/home/r112276/Med3Tab-PFN/sam-med3d/ckpt/sam_med3d_turbo.pth"

echo ""
echo "1. Checking checkpoint file existence..."
if [ -f "$CKPT_PATH" ]; then
    echo "   ✓ Checkpoint file exists"
    
    # Check file size
    SIZE=$(stat -c%s "$CKPT_PATH" 2>/dev/null || stat -f%z "$CKPT_PATH" 2>/dev/null)
    SIZE_MB=$((SIZE / 1024 / 1024))
    echo "   File size: ${SIZE_MB} MB"
    
    if [ $SIZE_MB -lt 700 ]; then
        echo "   ⚠️  WARNING: File is smaller than expected (should be ~750MB)"
        echo "   The download might be incomplete!"
    else
        echo "   ✓ File size looks correct"
    fi
    
    # Check file permissions
    if [ -r "$CKPT_PATH" ]; then
        echo "   ✓ File is readable"
    else
        echo "   ❌ ERROR: File is not readable!"
    fi
else
    echo "   ❌ ERROR: Checkpoint file NOT FOUND!"
    echo "   Expected location: $CKPT_PATH"
    echo ""
    echo "   Download it with:"
    echo "   cd $(dirname $CKPT_PATH)"
    echo "   wget https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth"
    exit 1
fi

echo ""
echo "2. Running Python verification..."
cd /trinity/home/r112276/Med3Tab-PFN
python cluster_scripts/verify_weights.py

echo ""
echo "=========================================="
echo "Verification complete!"
echo "=========================================="
