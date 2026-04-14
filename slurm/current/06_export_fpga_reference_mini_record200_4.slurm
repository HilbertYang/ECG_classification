#!/bin/bash
#------------------------------------------------------------
# SCRIPT 6: Export a tiny FPGA reference subset for record 200
#
# PURPOSE: Export a very small hardware-debug dataset using the
#          first 4 beats from MIT-BIH record 200, along with
#          software reference logits and predicted classes.
#
# PREREQUISITE:
#   - Run 01_setup_env.slurm first
#   - Train a model first so checkpoints/mitbih_lightweight.pt exists
#   - Have a processed dataset such as data/processed/mitbih_binary.npz
#
# HOW TO SUBMIT:
#   sbatch slurm/current/06_export_fpga_reference_mini_record200_4.slurm
#
# OUTPUT:
#   hardware_export/mini_feature/mini_mitbih_record_200_4beats_meta.json
#   hardware_export/mini_feature/mini_mitbih_record_200_4beats_features_bf16.mem
#   hardware_export/mini_feature/mini_mitbih_record_200_4beats_fp32_logits.txt
#   hardware_export/mini_feature/mini_mitbih_record_200_4beats_bf16_logits.txt
#   hardware_export/mini_feature/mini_mitbih_record_200_4beats_fp32_pred.txt
#   hardware_export/mini_feature/mini_mitbih_record_200_4beats_bf16_pred.txt
#------------------------------------------------------------

#SBATCH --job-name=mini200_4
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=00:10:00
#SBATCH --output=logs/current_export_fpga_reference_mini_record200_4.out
#SBATCH --error=logs/current_export_fpga_reference_mini_record200_4.err

set -e

echo "============================================"
echo "  Export Tiny FPGA Reference Set (Record 200, 4 Beats)"
echo "============================================"
echo "Started: $(date)"
echo "Node   : $SLURMD_NODENAME"
echo ""

cd "$SLURM_SUBMIT_DIR"

FULL_CKPT="checkpoints/mitbih_lightweight.pt"
DATASET="data/processed/mitbih_binary.npz"
OUT_DIR="hardware_export/mini_feature"
RECORD_ID="200"
BEAT_COUNT="4"
PREFIX="mini_mitbih_record_200_4beats"

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

mkdir -p "$OUT_DIR"

echo "--- Exporting record $RECORD_ID first $BEAT_COUNT beats ---"
python -m src.export_fpga_reference \
    --checkpoint "$FULL_CKPT" \
    --dataset "$DATASET" \
    --record-id "$RECORD_ID" \
    --first-n-beats "$BEAT_COUNT" \
    --accumulation fp32 \
    --out-dir "$OUT_DIR" \
    --prefix "$PREFIX"

echo ""
echo "============================================"
echo "  Exported mini FPGA reference files:"
ls -lh "$OUT_DIR"/"$PREFIX"*
echo "============================================"
echo "Finished: $(date)"
