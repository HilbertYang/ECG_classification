#!/bin/bash
#------------------------------------------------------------
# SCRIPT 3: Full GPU training on MIT-BIH dataset
#
# PURPOSE: Download the MIT-BIH dataset from PhysioNet,
#          prepare it, then train the model on a GPU.
#
# PREREQUISITE: Run 01_setup_env.slurm first.
#
# HOW TO SUBMIT:
#   sbatch slurm/current/03_gpu_train.slurm
#
# HOW TO MONITOR:
#   squeue --me                        # check if job is running
#   tail -f logs/current_gpu_train.out # watch output live
#
# HOW TO CANCEL:
#   scancel <JOB_ID>
#
# IMPORTANT — set your CARC account name below:
#   Replace "YOUR_ACCOUNT" with your PI's CARC allocation,
#   e.g. "ttrojan_123". Find yours with:  sacctmgr show user $USER
#------------------------------------------------------------

#SBATCH --job-name=ecg_curr_train
#SBATCH --partition=gpu                # GPU queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # DataLoader worker threads
#SBATCH --mem=32GB
#SBATCH --time=02:00:00               # 2-hour limit; adjust if needed
#SBATCH --gres=gpu:1                  # Request 1 GPU
##SBATCH --account=YOUR_ACCOUNT       # Uncomment and fill in your account

#SBATCH --output=logs/current_gpu_train.out
#SBATCH --error=logs/current_gpu_train.err

set -e

echo "============================================"
echo "  ECG GPU Training — MIT-BIH Dataset"
echo "============================================"
echo "Started : $(date)"
echo "Node    : $SLURMD_NODENAME"
echo "Job ID  : $SLURM_JOB_ID"
echo ""

cd "$SLURM_SUBMIT_DIR"

# Load and activate Conda environment
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate ecg

# --- Confirm GPU is visible ---
echo "--- GPU Info ---"
nvidia-smi
echo ""

# --- Step 1: Prepare the dataset ---
# This streams the MIT-BIH records from PhysioNet and saves a .npz file.
# It only needs to run once; if the file already exists, skip it.
DATASET="data/processed/mitbih_binary.npz"

if [ -f "$DATASET" ]; then
    echo "Dataset already exists at $DATASET — skipping preparation."
else
    echo "--- Preparing MIT-BIH dataset ---"
    mkdir -p data/processed
    python -m src.prepare_mitbih \
        --stream \
        --output "$DATASET"
    echo "Dataset saved to $DATASET"
fi
echo ""

# --- Step 2: Train ---
echo "--- Training ---"
python -m src.train \
    --dataset "$DATASET" \
    --run-name mitbih_lightweight \
    --epochs 20

echo ""

# --- Step 3: Evaluate on the test split ---
echo "--- Evaluation ---"
python -m src.evaluate \
    --checkpoint checkpoints/mitbih_lightweight.pt \
    --dataset "$DATASET" \
    --split test

echo ""

# --- Step 4: Single-sample inference example ---
echo "--- Inference (sample index 0) ---"
python -m src.inference \
    --checkpoint checkpoints/mitbih_lightweight.pt \
    --dataset "$DATASET" \
    --split test \
    --index 0

echo ""
echo "============================================"
echo "  Output files:"
ls -lh checkpoints/ results/
echo "============================================"
echo "Finished: $(date)"
