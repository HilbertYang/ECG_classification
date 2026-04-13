#!/bin/bash
#------------------------------------------------------------
# SCRIPT 10: Export fused frontend parameters for hardware
#
# PURPOSE: Export the current lightweight model into a hardware-
#          friendly parameter set where:
#            Conv1d + BatchNorm1d are fused into one equivalent
#            Conv1d, and the classifier is exported alongside it.
#
# PREREQUISITE:
#   - Run 01_setup_env.slurm first
#   - Train a model first so checkpoints/mitbih_lightweight.pt exists
#
# HOW TO SUBMIT:
#   sbatch slurm/current/10_export_fused_frontend_hw.slurm
#
# OUTPUT:
#   hardware_export/fused_frontend/ecg_fused_frontend_meta.json
#   hardware_export/fused_frontend/ecg_fused_frontend_conv_weight_float.txt
#   hardware_export/fused_frontend/ecg_fused_frontend_conv_bias_float.txt
#   hardware_export/fused_frontend/ecg_fused_frontend_conv_weight_q16_8.mem
#   hardware_export/fused_frontend/ecg_fused_frontend_conv_bias_q16_8.mem
#   hardware_export/fused_frontend/ecg_fused_frontend_classifier_weight_float.txt
#   hardware_export/fused_frontend/ecg_fused_frontend_classifier_bias_float.txt
#   hardware_export/fused_frontend/ecg_fused_frontend_classifier_weight_q16_8.mem
#   hardware_export/fused_frontend/ecg_fused_frontend_classifier_bias_q16_8.mem
#------------------------------------------------------------

#SBATCH --job-name=fused_curr
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=00:10:00
#SBATCH --output=logs/current_export_fused_frontend_hw.out
#SBATCH --error=logs/current_export_fused_frontend_hw.err

set -e

echo "============================================"
echo "  Export Fused Frontend Parameters"
echo "============================================"
echo "Started: $(date)"
echo "Node   : $SLURMD_NODENAME"
echo ""

cd "$SLURM_SUBMIT_DIR"

FULL_CKPT="checkpoints/mitbih_lightweight.pt"
OUT_DIR="hardware_export/fused_frontend"
PREFIX="ecg_fused_frontend"

if [ ! -f "$FULL_CKPT" ]; then
    echo "ERROR: Full checkpoint not found: $FULL_CKPT"
    echo "Train the model first:"
    echo "  sbatch slurm/current/03_gpu_train.slurm"
    exit 1
fi

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate ecg

echo "Python: $(which python)"
echo ""

mkdir -p "$OUT_DIR"

echo "--- Exporting fused Conv+BN frontend and classifier parameters ---"
python -m src.export_fused_frontend_hw \
    --checkpoint "$FULL_CKPT" \
    --out-dir "$OUT_DIR" \
    --prefix "$PREFIX" \
    --word-bits 16 \
    --frac-bits 8

echo ""
echo "============================================"
echo "  Exported fused hardware files:"
ls -lh "$OUT_DIR"/"$PREFIX"*
echo "============================================"
echo "Finished: $(date)"
