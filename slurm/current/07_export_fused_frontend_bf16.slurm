#!/bin/bash
#------------------------------------------------------------
# SCRIPT 7: Export fused frontend parameters as BF16
#
# PURPOSE: Export the current lightweight model into BF16
#          hardware parameter files where:
#            Conv1d + BatchNorm1d are fused into one equivalent
#            Conv1d, and the classifier is exported alongside it.
#
# HOW TO SUBMIT:
#   sbatch slurm/current/07_export_fused_frontend_bf16.slurm
#------------------------------------------------------------

#SBATCH --job-name=fused_bf16
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=00:10:00
#SBATCH --output=logs/current_export_fused_frontend_bf16.out
#SBATCH --error=logs/current_export_fused_frontend_bf16.err

set -e

cd "$SLURM_SUBMIT_DIR"

FULL_CKPT="checkpoints/mitbih_lightweight.pt"
OUT_DIR="hardware_export/fused_frontend_bf16"
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

mkdir -p "$OUT_DIR"

python -m src.export_fused_frontend_bf16 \
    --checkpoint "$FULL_CKPT" \
    --out-dir "$OUT_DIR" \
    --prefix "$PREFIX"

ls -lh "$OUT_DIR"/"$PREFIX"*
