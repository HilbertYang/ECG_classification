#!/bin/bash
#------------------------------------------------------------
# SCRIPT 7: Emulate BF16 classifier and compare with FP32
#
# PURPOSE: Run a software-side BF16 emulation of the classifier
#          before hardware deployment and measure the difference
#          from the original FP32 classifier.
#
# PREREQUISITE:
#   - Run 01_setup_env.slurm first
#   - Train a model first so checkpoints/mitbih_lightweight.pt exists
#   - Have a processed dataset such as data/processed/mitbih_binary.npz
#
# HOW TO SUBMIT:
#   sbatch slurm/current/07_emulate_classifier_bf16.slurm
#
# OUTPUT:
#   results/mitbih_lightweight_test_bf16_fp32.json
#------------------------------------------------------------

#SBATCH --job-name=emu_curr
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:20:00
#SBATCH --output=logs/current_emulate_classifier_bf16.out
#SBATCH --error=logs/current_emulate_classifier_bf16.err

set -e

echo "============================================"
echo "  Emulate BF16 Classifier"
echo "============================================"
echo "Started: $(date)"
echo "Node   : $SLURMD_NODENAME"
echo ""

cd "$SLURM_SUBMIT_DIR"

FULL_CKPT="checkpoints/mitbih_lightweight.pt"
DATASET="data/processed/mitbih_binary.npz"

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

echo "Python: $(which python)"
echo ""

echo "--- Running BF16 emulation with FP32 accumulation ---"
python -m src.emulate_classifier_bf16 \
    --checkpoint "$FULL_CKPT" \
    --dataset "$DATASET" \
    --split test \
    --accumulation fp32

echo ""
echo "--- Running BF16 emulation with BF16 accumulation ---"
python -m src.emulate_classifier_bf16 \
    --checkpoint "$FULL_CKPT" \
    --dataset "$DATASET" \
    --split test \
    --accumulation bf16

echo ""
echo "============================================"
echo "  BF16 emulation reports:"
ls -lh results/*bf16*
echo "============================================"
echo "Finished: $(date)"
