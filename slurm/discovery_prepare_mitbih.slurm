#!/bin/bash

# Submit with:
#   sbatch -A <project_id> slurm/discovery_prepare_mitbih.slurm
#
# Optional overrides:
#   CONDA_ENV=ecg
#   OUTPUT=data/processed/mitbih_binary.npz
#   RECORDS="100 101 102"

#SBATCH --job-name=ecg-prepare
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_ENV="${CONDA_ENV:-ecg}"
OUTPUT="${OUTPUT:-data/processed/mitbih_binary.npz}"
RECORDS="${RECORDS:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

module purge

if [[ -n "${CONDA_ENV}" ]]; then
    module load conda
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
elif [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
else
    echo "No Conda environment configured and no .venv found."
    echo "Set CONDA_ENV=<env_name> when submitting, or create .venv in the repo."
    exit 1
fi

prep_cmd=(
    python -m src.prepare_mitbih
    --download
    --output "${OUTPUT}"
)

if [[ -n "${RECORDS}" ]]; then
    read -r -a record_array <<< "${RECORDS}"
    prep_cmd+=(--records "${record_array[@]}")
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Working directory: ${SLURM_SUBMIT_DIR}"
echo "Launching dataset preparation:"
printf ' %q' "${prep_cmd[@]}"
echo

srun "${prep_cmd[@]}"
