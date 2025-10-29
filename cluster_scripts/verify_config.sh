#!/bin/bash
# Quick verification script - check config before running

CONFIG_FILE="${1:-configs/datasets_cluster.yaml}"

echo "=========================================="
echo "Config Verification"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo ""

# Check if file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found!"
    exit 1
fi

echo "✓ Config file exists"
echo ""

# Extract datasets using Python
python << EOF
import yaml
import sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        cfg = yaml.safe_load(f)
    
    if 'datasets' not in cfg:
        print("❌ No 'datasets' key found in config!")
        sys.exit(1)
    
    datasets = cfg['datasets']
    print(f"📊 Found {len(datasets)} datasets in config:\n")
    
    for i, (name, config) in enumerate(datasets.items(), 1):
        root = config.get('dataset_root', 'NOT SET')
        ds_name = config.get('labels', {}).get('dataset_name', 'NOT SET')
        print(f"  {i}. {name}")
        print(f"     Path: {root}")
        print(f"     Dataset name in sheet: {ds_name}")
        print()
    
    print("✅ Config is valid!")
    
except Exception as e:
    print(f"❌ Error reading config: {e}")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "Next step: Run dry-run"
echo "=========================================="
echo "python cluster_scripts/run_experiment1_benchmarks.py \\"
echo "    --config $CONFIG_FILE \\"
echo "    --dry-run"
echo ""
