#!/bin/bash
#------------------------------------------------------------
# SCRIPT 8: Export CNN features and software reference for FPGA
#
# PURPOSE: Export the CNN feature vectors that will be fed into
#          FPGA logic together with software reference logits
#          and predicted classes for hardware comparison.
#
# PREREQUISITE:
#   - Run 01_setup_env.slurm first
#   - Train a model first so checkpoints/mitbih_lightweight.pt exists
#   - Have a processed dataset such as data/processed/mitbih_binary.npz
#
# HOW TO SUBMIT:
#   sbatch slurm/current/08_export_fpga_reference.slurm
#
# OUTPUT:
#   hardware_export/extracted_feature/mitbih_lightweight_test_meta.json
#   hardware_export/extracted_feature/mitbih_lightweight_test_features_fp32.txt
#   hardware_export/extracted_feature/mitbih_lightweight_test_features_bf16.mem
#   hardware_export/extracted_feature/mitbih_lightweight_test_labels.txt
#   hardware_export/extracted_feature/mitbih_lightweight_test_fp32_logits.txt
#   hardware_export/extracted_feature/mitbih_lightweight_test_bf16_logits.txt
#   hardware_export/extracted_feature/mitbih_lightweight_test_fp32_pred.txt
#   hardware_export/extracted_feature/mitbih_lightweight_test_bf16_pred.txt
#   hardware_export/extracted_feature/mitbih_lightweight_test_reference_bundle.npz
#------------------------------------------------------------

#SBATCH --job-name=fpga_curr
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:20:00
#SBATCH --output=logs/current_export_fpga_reference.out
#SBATCH --error=logs/current_export_fpga_reference.err

set -e

echo "============================================"
echo "  Export FPGA Inputs and Reference Outputs"
echo "============================================"
echo "Started: $(date)"
echo "Node   : $SLURMD_NODENAME"
echo ""

cd "$SLURM_SUBMIT_DIR"

FULL_CKPT="checkpoints/mitbih_lightweight.pt"
DATASET="data/processed/mitbih_binary.npz"
OUT_DIR="hardware_export/extracted_feature"

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

echo "--- Exporting test split CNN features and software references ---"
python -m src.export_fpga_reference \
    --checkpoint "$FULL_CKPT" \
    --dataset "$DATASET" \
    --split test \
    --accumulation fp32 \
    --out-dir "$OUT_DIR"

echo ""
echo "============================================"
echo "  Exported FPGA reference files:"
ls -lh "$OUT_DIR"/mitbih_lightweight_test*
echo "============================================"
echo "Finished: $(date)"
