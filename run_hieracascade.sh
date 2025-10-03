#!/bin/bash
# Quick start script for HieraCascade training
# 
# Usage:
#   bash run_hieracascade.sh [fold_number] [label_column]
#
# Examples:
#   bash run_hieracascade.sh 0 Diagnosis
#   bash run_hieracascade.sh 0 Diagnosis_binary

FOLD=${1:-0}
LABEL_COL=${2:-Diagnosis}
DATA_ROOT="data"
SHEET_CSV="data/sheet.csv"
OUTPUT_DIR="outputs/hieracascade"

echo "========================================"
echo "HieraCascade Training Pipeline"
echo "========================================"
echo "Fold: $FOLD"
echo "Label column: $LABEL_COL"
echo "Data root: $DATA_ROOT"
echo "Sheet CSV: $SHEET_CSV"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Check if Python and required packages are available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8+"
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q -r hieracascade/requirements.txt

# Run the pipeline
echo ""
echo "Starting training pipeline..."
python -m hieracascade.quick_start \
    --data_root "$DATA_ROOT" \
    --sheet_csv "$SHEET_CSV" \
    --label_column "$LABEL_COL" \
    --output_dir "$OUTPUT_DIR" \
    --fold "$FOLD" \
    --device cuda

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. View training curves: $OUTPUT_DIR/stage2/fold$FOLD/plots/training_curves.png"
echo "2. View saliency maps: $OUTPUT_DIR/stage1/fold$FOLD/visualizations/"
echo "3. Evaluate model:"
echo "   python -m hieracascade.evaluate \\"
echo "     --checkpoint $OUTPUT_DIR/stage2/fold$FOLD/checkpoint_best.pt \\"
echo "     --stage stage2 \\"
echo "     --data_root $DATA_ROOT \\"
echo "     --labels_csv $OUTPUT_DIR/labels.csv \\"
echo "     --stage1_ckpt $OUTPUT_DIR/stage1/fold$FOLD/checkpoint_best.pt \\"
echo "     --output_dir $OUTPUT_DIR/stage2/fold$FOLD/eval \\"
echo "     --fold $FOLD"
