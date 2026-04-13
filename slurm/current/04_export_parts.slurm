#!/bin/bash
#------------------------------------------------------------
# SCRIPT 4: Export trained model into two separate .pt files
#
# PURPOSE: Split a trained ECG checkpoint into:
#   1) feature extractor
#   2) classifier
#
# PREREQUISITE:
#   - Run 01_setup_env.slurm first
#   - Train a model first so checkpoints/mitbih_lightweight.pt exists
#
# HOW TO SUBMIT:
#   sbatch slurm/current/04_export_parts.slurm
#
# OUTPUT:
#   checkpoints/mitbih_lightweight_feature_extractor.pt
#   checkpoints/mitbih_lightweight_classifier.pt
#------------------------------------------------------------

#SBATCH --job-name=parts_curr
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=00:10:00
#SBATCH --output=logs/current_export_parts.out
#SBATCH --error=logs/current_export_parts.err

set -e

echo "============================================"
echo "  Export ECG Model Parts"
echo "============================================"
echo "Started: $(date)"
echo "Node   : $SLURMD_NODENAME"
echo ""

cd "$SLURM_SUBMIT_DIR"

FULL_CKPT="checkpoints/mitbih_lightweight.pt"

if [ ! -f "$FULL_CKPT" ]; then
    echo "ERROR: Full checkpoint not found: $FULL_CKPT"
    echo "Train the model first, for example with:"
    echo "  sbatch slurm/current/03_gpu_train.slurm"
    exit 1
fi

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate ecg

echo "Python: $(which python)"
echo ""

echo "--- Exporting feature extractor and classifier ---"
python -m src.export_parts --checkpoint "$FULL_CKPT"

echo ""
echo "============================================"
echo "  Exported files:"
ls -lh checkpoints/*feature_extractor.pt checkpoints/*classifier.pt
echo "============================================"
echo "Finished: $(date)"
