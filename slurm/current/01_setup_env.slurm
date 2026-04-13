#!/bin/bash
#------------------------------------------------------------
# SCRIPT 1: Set up Conda environment
#
# PURPOSE: Create a Conda environment named "ecg" and install
#          all packages from requirements.txt.
#          Run this ONCE before any training jobs.
#
# HOW TO SUBMIT:
#   sbatch slurm/current/01_setup_env.slurm
#
# HOW TO READ OUTPUT:
#   cat logs/current_setup_env.out
#
# NOTE: This can take 5-10 minutes because pip downloads
#       PyTorch and other large packages.
#------------------------------------------------------------

#SBATCH --job-name=setup_curr
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # More CPUs = faster pip install
#SBATCH --mem=8GB
#SBATCH --time=00:30:00                # Give it 30 min to download/install
#SBATCH --output=logs/current_setup_env.out
#SBATCH --error=logs/current_setup_env.err

# Exit immediately if any command fails
# This way the log will clearly show where things went wrong
set -e

echo "============================================"
echo "  Setting up Conda environment"
echo "============================================"
echo "Started: $(date)"
echo ""

# Go to the project root (where you ran sbatch from)
cd "$SLURM_SUBMIT_DIR"
echo "Working in: $(pwd)"
echo ""

# Load Conda from the CARC module system
module purge
module load conda
eval "$(conda shell.bash hook)"
echo "Conda: $(which conda)"
echo ""

# --- Create the Conda environment ---
# This environment is isolated to this project workflow.
if conda env list | awk '{print $1}' | grep -qx "ecg"; then
    echo "Conda environment 'ecg' already exists — skipping creation."
else
    echo "Creating Conda environment 'ecg' with Python 3.11 ..."
    conda create -n ecg python=3.11 -y
    echo "Done."
fi
echo ""

# --- Activate the Conda environment ---
echo "Activating Conda environment 'ecg' ..."
conda activate ecg
echo "Python is now: $(which python)"
echo "Python version: $(python --version)"
echo ""

# --- Upgrade pip first (good practice) ---
echo "Upgrading pip ..."
pip install --upgrade pip --quiet
echo ""

# --- Install all project dependencies ---
echo "Installing packages from requirements.txt ..."
pip install -r requirements.txt

echo ""
echo "============================================"
echo "  Installed packages:"
pip list
echo "============================================"
echo ""
echo "Setup complete at: $(date)"
echo ""
echo "NEXT STEP: Run the current-version demo training job:"
echo "  sbatch slurm/current/02_demo_train.slurm"
