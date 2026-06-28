#!/bin/bash
# FILENAME: tempscale.sh
# SLURM worker for tempscale eval. Dispatches to per-arch tempscale_eval.py.

#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=${NODE}
#SBATCH --time=0:29:59

module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12

PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

export PYTHONUNBUFFERED=1

if [ -z "${ARCH}" ] || [ -z "${FOLD_ID}" ] || [ -z "${CKPT_DIR}" ]; then
    echo "ERROR: ARCH, FOLD_ID, CKPT_DIR must be set via --export"
    exit 1
fi

echo "==========================================="
echo "Tempscale Eval"
echo "  ARCH:     ${ARCH}"
echo "  FOLD_ID:  ${FOLD_ID}"
echo "  CKPT_DIR: ${CKPT_DIR}"
echo "==========================================="

case ${ARCH} in
    mamnet)
        cd ${PROJECT_ROOT}/python/mamnet
        $PYTHON_BIN -u tempscale_eval.py \
            --checkpoint_dir ${CKPT_DIR} \
            --base_data_root ${BASE_DATA_ROOT} \
            --resolution highres \
            --fold_id ${FOLD_ID}
        ;;
    oglanet)
        cd ${PROJECT_ROOT}/python/oglanet
        $PYTHON_BIN -u tempscale_eval.py \
            --checkpoint_dir ${CKPT_DIR} \
            --base_data_root ${BASE_DATA_ROOT} \
            --resolution highres \
            --fold_id ${FOLD_ID}
        ;;
    dinov3)
        cd ${PROJECT_ROOT}/python/dinov3
        $PYTHON_BIN -u tempscale_eval.py \
            --checkpoint_dir ${CKPT_DIR} \
            --base_data_root ${BASE_DATA_ROOT} \
            --resolution highres \
            --fold_id ${FOLD_ID} \
            --weights_path ${WEIGHT_DIR_DINOV3}
        ;;
    *)
        echo "ERROR: unknown ARCH=${ARCH}"
        exit 1
        ;;
esac