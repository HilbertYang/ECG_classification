#!/bin/bash
#------------------------------------------------------------
# SCRIPT 2: Demo training (no dataset needed, CPU only)
#
# PURPOSE: Train the ECG model on *synthetic* random data
#          just to verify the code runs end-to-end.
#          Great for testing before you have the real dataset.
#
# PREREQUISITE: Run 01_setup_env.slurm first.
#
# HOW TO SUBMIT:
#   sbatch slurm/02_demo_train.slurm
#
# HOW TO READ OUTPUT:
#   cat logs/demo_train.out
#
# WHAT TO EXPECT:
#   Training loss printed each epoch, then a checkpoint saved
#   to checkpoints/demo_baseline.pt
#------------------------------------------------------------

#SBATCH --job-name=demo_train
#SBATCH --partition=main               # CPU queue — no GPU needed for demo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:15:00
#SBATCH --output=logs/demo_train.out
#SBATCH --error=logs/demo_train.err

set -e

echo "============================================"
echo "  ECG Demo Training"
echo "============================================"
echo "Started: $(date)"
echo "Node   : $SLURMD_NODENAME"
echo ""

cd "$SLURM_SUBMIT_DIR"

# Activate the Conda environment we created in step 1
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate ecg
echo "Python: $(which python)"
echo ""

# --- Run the demo training ---
# --demo        : use synthetic data instead of a real dataset file
# --run-name    : names the output files (checkpoint + results JSON)
# --epochs 10   : train for 10 passes over the data
echo "Starting training ..."
python -m src.train \
    --demo \
    --run-name demo_baseline \
    --epochs 10

echo ""
echo "--- Evaluating on demo data ---"
python -m src.evaluate \
    --checkpoint checkpoints/demo_baseline.pt \
    --demo

echo ""
echo "--- Running single-sample inference ---"
python -m src.inference \
    --checkpoint checkpoints/demo_baseline.pt \
    --demo \
    --index 0

echo ""
echo "============================================"
echo "  Output files created:"
ls -lh checkpoints/ results/
echo "============================================"
echo "Finished: $(date)"
echo ""
echo "NEXT STEP: For real data, run:"
echo "  sbatch slurm/03_gpu_train.slurm"
