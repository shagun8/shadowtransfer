#!/bin/bash
# FILENAME: build_loco_splits.sh
#
# Worker script for materializing one LOCO fold + resolution.
# Submitted by build_loco_splits_submit.sh.
#
# Inputs (from --export):
#   BASE_DATA_ROOT    : per-city dataset root
#   OUTPUT_ROOT       : where to write the LOCO tree
#   FOLD_ID           : 0 | 1 | 2
#   RESOLUTION        : highres | midres
#   TRANSFER_MODE     : copy (default) | symlink | hardlink
#   N_TRAIN_PER_CITY  : int  (default 225)
#   N_VAL_PER_CITY    : int  (default  75)
#   NO_MULTICLASS     : 1 to skip masks_multiclass/, else include it
#   SEED              : int  (default 42)
#
# This is an I/O-bound job (a few thousand small file copies). The SLURM
# block below mirrors the GPU partition the rest of the project uses so
# the same account is charged; swap to a CPU partition if your site has
# one and you'd rather not burn GPU allocation on file-copy work.

# ---- SBATCH: NCSA Delta ----
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00

# ---- SBATCH: Gilbreth (uncomment if running there) ----
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=standby
# #SBATCH --constraint=a100
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=8G
# #SBATCH --time=00:45:00

# ---- SBATCH: Anvil (uncomment if running there) ----
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=8G
# #SBATCH --time=00:25:00

# ---- Modules: NCSA Delta ----
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/mamnet
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

# ---- Modules: Gilbreth (uncomment if running there) ----
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# cd ${PROJECT_ROOT}/python/mamnet
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

# ---- Modules: Anvil (uncomment if running there) ----
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12
# cd ${PROJECT_ROOT}/python/mamnet
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python

# Sensible defaults for env vars that may be unset
: "${TRANSFER_MODE:=copy}"
: "${N_TRAIN_PER_CITY:=225}"
: "${N_VAL_PER_CITY:=75}"
: "${SEED:=42}"

NO_MULTICLASS_FLAG=""
if [ "${NO_MULTICLASS}" == "1" ]; then
    NO_MULTICLASS_FLAG="--no_multiclass"
fi

echo "==============================="
echo "Build LOCO Split"
echo "  BASE_DATA_ROOT : ${BASE_DATA_ROOT}"
echo "  OUTPUT_ROOT    : ${OUTPUT_ROOT}"
echo "  FOLD_ID        : ${FOLD_ID}"
echo "  RESOLUTION     : ${RESOLUTION}"
echo "  TRANSFER_MODE  : ${TRANSFER_MODE}"
echo "  N_TRAIN/CITY   : ${N_TRAIN_PER_CITY}"
echo "  N_VAL/CITY     : ${N_VAL_PER_CITY}"
echo "  NO_MULTICLASS  : ${NO_MULTICLASS:-0}"
echo "  SEED           : ${SEED}"
echo "==============================="

$PYTHON_BIN build_loco_splits.py \
    --base_data_root   ${BASE_DATA_ROOT} \
    --output_root      ${OUTPUT_ROOT} \
    --resolutions      ${RESOLUTION} \
    --folds            ${FOLD_ID} \
    --mode             ${TRANSFER_MODE} \
    --n_train_per_city ${N_TRAIN_PER_CITY} \
    --n_val_per_city   ${N_VAL_PER_CITY} \
    --seed             ${SEED} \
    ${NO_MULTICLASS_FLAG}