#!/bin/bash
#------------------------------------------------------------
# SCRIPT 6: Export classifier into BF16 memory files
#
# PURPOSE: Convert the trained classifier head into BF16 files
#          for a BF16-capable NetFPGA / custom GPU path.
#
# PREREQUISITE:
#   - Run 01_setup_env.slurm first
#   - Run 04_export_parts.slurm first, or otherwise make sure
#     checkpoints/mitbih_lightweight_classifier.pt exists
#
# HOW TO SUBMIT:
#   sbatch slurm/current/06_export_classifier_bf16.slurm
#
# OUTPUT:
#   hardware_export/param_bf16/netfpga_classifier_bf16_meta.json
#   hardware_export/param_bf16/netfpga_classifier_weight_bf16_float.txt
#   hardware_export/param_bf16/netfpga_classifier_bias_bf16_float.txt
#   hardware_export/param_bf16/netfpga_classifier_weight_bf16.mem
#   hardware_export/param_bf16/netfpga_classifier_bias_bf16.mem
#------------------------------------------------------------

#SBATCH --job-name=bf16_curr
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=00:10:00
#SBATCH --output=logs/current_export_classifier_bf16.out
#SBATCH --error=logs/current_export_classifier_bf16.err

set -e

echo "============================================"
echo "  Export Classifier as BF16"
echo "============================================"
echo "Started: $(date)"
echo "Node   : $SLURMD_NODENAME"
echo ""

cd "$SLURM_SUBMIT_DIR"

CLASSIFIER_CKPT="checkpoints/mitbih_lightweight_classifier.pt"
OUT_DIR="hardware_export/param_bf16"

if [ ! -f "$CLASSIFIER_CKPT" ]; then
    echo "ERROR: Classifier checkpoint not found: $CLASSIFIER_CKPT"
    echo "Run the split export first:"
    echo "  sbatch slurm/current/04_export_parts.slurm"
    exit 1
fi

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate ecg

echo "Python: $(which python)"
echo ""

mkdir -p "$OUT_DIR"

echo "--- Exporting BF16 classifier files ---"
python -m src.export_classifier_bf16 \
    --classifier-checkpoint "$CLASSIFIER_CKPT" \
    --out-dir "$OUT_DIR" \
    --prefix netfpga_classifier \
    --layout row-major

echo ""
echo "============================================"
echo "  Exported BF16 files:"
ls -lh "$OUT_DIR"/*bf16*
echo "============================================"
echo "Finished: $(date)"
