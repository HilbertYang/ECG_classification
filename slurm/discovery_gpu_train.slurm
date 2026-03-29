#!/bin/bash

# Submit with:
#   sbatch -A <project_id> slurm/discovery_gpu_train.slurm
#
# Optional overrides:
#   CONDA_ENV=ecg
#   DATASET=data/processed/mitbih_binary.npz
#   RUN_NAME=my_run
#   EPOCHS=10
#   BATCH_SIZE=64

#SBATCH --job-name=ecg-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_ENV="${CONDA_ENV:-ecg}"
DATASET="${DATASET:-data/processed/mitbih_binary.npz}"
RUN_NAME="${RUN_NAME:-discovery_gpu_${SLURM_JOB_ID}}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

module purge

if [[ -n "${CONDA_ENV}" ]]; then
    module load conda
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
elif [[ -f ".venv/bin/activate" ]]; then
    # Fallback for a repo-local virtualenv created on Discovery.
    source .venv/bin/activate
else
    echo "No Conda environment configured and no .venv found."
    echo "Set CONDA_ENV=<env_name> when submitting, or create .venv in the repo."
    exit 1
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Working directory: ${SLURM_SUBMIT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
PY

train_cmd=(
    python -m src.train
    --run-name "${RUN_NAME}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
)

if [[ -f "${DATASET}" ]]; then
    echo "Using dataset: ${DATASET}"
    train_cmd+=(--dataset "${DATASET}")
else
    echo "Dataset not found at ${DATASET}; running demo mode instead."
    train_cmd+=(--demo)
fi

echo "Launching training:"
printf ' %q' "${train_cmd[@]}"
echo

srun "${train_cmd[@]}"
