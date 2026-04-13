#!/bin/bash
#------------------------------------------------------------
# SCRIPT 13: Export record 200 first 4 input beats as BF16
#
# PURPOSE: Export the raw model input windows for the first
#          4 beats from MIT-BIH record 200 as BF16 files.
#
# HOW TO SUBMIT:
#   sbatch slurm/current/13_export_input_bf16_record200_4.slurm
#------------------------------------------------------------

#SBATCH --job-name=input200_4
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=00:10:00
#SBATCH --output=logs/current_export_input_bf16_record200_4.out
#SBATCH --error=logs/current_export_input_bf16_record200_4.err

set -e

cd "$SLURM_SUBMIT_DIR"

FULL_CKPT="checkpoints/mitbih_lightweight.pt"
DATASET="data/processed/mitbih_binary.npz"
OUT_DIR="hardware_export/input_bf16"
PREFIX="mini_mitbih_record_200_4beats_input"

if [ ! -f "$FULL_CKPT" ]; then
    echo "ERROR: Full checkpoint not found: $FULL_CKPT"
    echo "Train the model first:"
    echo "  sbatch slurm/current/03_gpu_train.slurm"
    exit 1
fi

if [ ! -f "$DATASET" ]; then
    echo "ERROR: Processed dataset not found: $DATASET"
    echo "Prepare the dataset first or rerun:"
    echo "  sbatch slurm/current/03_gpu_train.slurm"
    exit 1
fi

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate ecg

mkdir -p "$OUT_DIR"

python -m src.export_input_bf16 \
    --checkpoint "$FULL_CKPT" \
    --dataset "$DATASET" \
    --record-id 200 \
    --first-n-beats 4 \
    --out-dir "$OUT_DIR" \
    --prefix "$PREFIX"

ls -lh "$OUT_DIR"/"$PREFIX"*
