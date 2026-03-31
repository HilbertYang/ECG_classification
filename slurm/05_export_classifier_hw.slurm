#!/bin/bash
#------------------------------------------------------------
# SCRIPT 5: Export classifier into FPGA-friendly files
#
# PURPOSE: Convert the trained classifier head into:
#   - human-readable float files
#   - fixed-point .mem files for FPGA loading
#
# PREREQUISITE:
#   - Run 01_setup_env.slurm first
#   - Run 04_export_parts.slurm first, or otherwise make sure
#     checkpoints/mitbih_baseline_classifier.pt exists
#
# HOW TO SUBMIT:
#   sbatch slurm/05_export_classifier_hw.slurm
#
# OUTPUT:
#   hardware_export/param_fp32/netfpga_classifier_meta.json
#   hardware_export/param_fp32/netfpga_classifier_weight_matrix.txt
#   hardware_export/param_fp32/netfpga_classifier_bias_float.txt
#   hardware_export/param_fp32/netfpga_classifier_weight_q16_8.mem
#   hardware_export/param_fp32/netfpga_classifier_bias_q16_8.mem
#------------------------------------------------------------

#SBATCH --job-name=export_hw
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=00:10:00
#SBATCH --output=logs/export_classifier_hw.out
#SBATCH --error=logs/export_classifier_hw.err

set -e

echo "============================================"
echo "  Export Classifier for NetFPGA"
echo "============================================"
echo "Started: $(date)"
echo "Node   : $SLURMD_NODENAME"
echo ""

cd "$SLURM_SUBMIT_DIR"

CLASSIFIER_CKPT="checkpoints/mitbih_baseline_classifier.pt"
OUT_DIR="hardware_export/param_fp32"

if [ ! -f "$CLASSIFIER_CKPT" ]; then
    echo "ERROR: Classifier checkpoint not found: $CLASSIFIER_CKPT"
    echo "Run the split export first:"
    echo "  sbatch slurm/04_export_parts.slurm"
    exit 1
fi

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate ecg

echo "Python: $(which python)"
echo ""

mkdir -p "$OUT_DIR"

echo "--- Exporting FPGA-friendly classifier files ---"
python -m src.export_classifier_hw \
    --classifier-checkpoint "$CLASSIFIER_CKPT" \
    --out-dir "$OUT_DIR" \
    --prefix netfpga_classifier \
    --layout row-major \
    --word-bits 16 \
    --frac-bits 8

echo ""
echo "============================================"
echo "  Exported hardware files:"
ls -lh "$OUT_DIR"
echo "============================================"
echo "Finished: $(date)"
