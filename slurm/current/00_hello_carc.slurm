#!/bin/bash
#------------------------------------------------------------
# SCRIPT 0: Hello CARC — your first SLURM job
#
# PURPOSE: Verify that SLURM works, check your environment,
#          and make sure Python is available.
#
# HOW TO SUBMIT:
#   sbatch slurm/current/00_hello_carc.slurm
#
# HOW TO CHECK STATUS:
#   squeue --me
#
# HOW TO READ OUTPUT:
#   cat logs/current_hello_carc.out
#------------------------------------------------------------

# --- SLURM directives (lines starting with #SBATCH) ---
# These tell the scheduler what resources you need.

#SBATCH --job-name=hello_curr          # Name shown in squeue
#SBATCH --partition=main               # Queue to submit to (main = general CPU)
#SBATCH --nodes=1                      # Number of compute nodes
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=1              # CPU cores per task
#SBATCH --mem=4GB                      # RAM
#SBATCH --time=00:05:00                # Max wall time (HH:MM:SS)
#SBATCH --output=logs/current_hello_carc.out   # Where stdout goes
#SBATCH --error=logs/current_hello_carc.err    # Where stderr goes

# --- Your shell commands start here ---

echo "============================================"
echo "  Hello from CARC Discovery!"
echo "============================================"
echo ""

echo "--- Job Info ---"
echo "Job ID     : $SLURM_JOB_ID"
echo "Job Name   : $SLURM_JOB_NAME"
echo "Node       : $SLURMD_NODENAME"
echo "Submitted  : $(date)"
echo ""

echo "--- System Info ---"
echo "Hostname   : $(hostname)"
echo "OS         : $(uname -a)"
echo "CPU cores  : $(nproc)"
echo "RAM (free) : $(free -h | awk '/^Mem:/ {print $4}')"
echo ""

echo "--- Python Check ---"
# Load the Python module that CARC provides
module load python/3.11.3

python3 --version
echo "Python is at: $(which python3)"
echo ""

echo "--- Project Directory ---"
# $SLURM_SUBMIT_DIR is where you ran sbatch from
echo "Working dir: $SLURM_SUBMIT_DIR"
ls "$SLURM_SUBMIT_DIR"
echo ""

echo "Job finished at: $(date)"
echo "============================================"
