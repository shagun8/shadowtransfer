#!/bin/bash
# FILENAME: split_diag.sh
#
# SLURM job script for split distribution diagnostics.
# Called by split_diag_submit.sh — do not run directly.

# ---- SBATCH: Server-specific (uncomment the one you need) ----

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:59:59

# --- Gilbreth ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=standby
# #SBATCH --constraint=a100
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=64G
# #SBATCH --time=2:59:59

# --- Anvil ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=64G
# #SBATCH --time=2:59:59


# ---- Modules (uncomment the one you need) ----

# --- NCSA Delta ---
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

# --- Gilbreth ---
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# cd ${PROJECT_ROOT}/python
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

# --- Anvil ---
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12
# cd ${PROJECT_ROOT}/python
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python


# ---- Ensure scikit-learn is available ----
$PYTHON_BIN -m pip install scikit-learn --quiet 2>/dev/null || true

# ---- Build optional flags ----
GEO_FLAG=""
if [ -n "${GEO_METADATA_PATH}" ]; then
    GEO_FLAG="--geo_metadata_path ${GEO_METADATA_PATH}"
fi

SKIP_FEATURES_FLAG=""
if [ "${SKIP_FEATURES}" == "1" ]; then
    SKIP_FEATURES_FLAG="--skip_features"
fi

SKIP_CROSS_CITY_FLAG=""
if [ "${SKIP_CROSS_CITY}" == "1" ]; then
    SKIP_CROSS_CITY_FLAG="--skip_cross_city"
fi

echo "============================================"
echo "Split Distribution Diagnostics"
echo "============================================"
echo "Base data root:  ${BASE_DATA_ROOT}"
echo "Resolutions:     ${RESOLUTIONS}"
echo "Cities:          ${CITIES}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "Geo metadata:    ${GEO_FLAG}"
echo "Skip features:   ${SKIP_FEATURES_FLAG}"
echo "Batch size:      ${BATCH_SIZE:-32}"
echo "============================================"

# ---- Run ----
$PYTHON_BIN split_diagnostics.py \
    --base_data_root ${BASE_DATA_ROOT} \
    --resolutions ${RESOLUTIONS} \
    --cities ${CITIES} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE:-32} \
    --device cuda \
    ${GEO_FLAG} \
    ${SKIP_FEATURES_FLAG} \
    ${SKIP_CROSS_CITY_FLAG}

echo "Done!"