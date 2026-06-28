#!/bin/bash
# FILENAME: tempscale_aggregate.sh
# Runs the §5 case study aggregator. CPU-only — no GPU needed.

#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=${NODE}
#SBATCH --time=0:14:59

module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12

PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

cd ${PROJECT_ROOT}/python

export PYTHONUNBUFFERED=1

$PYTHON_BIN -u tempscale_aggregate.py \
    --mamnet_root  ${PROJECT_ROOT}/data/mamnet/outputs \
    --oglanet_root ${PROJECT_ROOT}/data/oglanet/outputs \
    --dinov3_root  ${PROJECT_ROOT}/data/dinov3/outputs \
    --output_dir   ${PROJECT_ROOT}/data/case_study_output \
    --boundary_tolerance 2 \
    --n_bootstrap 10000