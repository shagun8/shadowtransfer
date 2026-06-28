#!/bin/bash
# FILENAME: loco_eval.sh
# SLURM script for running LOCO cross-city evaluation

#SBATCH -A <SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3:59:59

# Navigate to project directory
cd ${PROJECT_ROOT}/python/oglanet

# Load required modules
module load conda
module load cuda/12.1.1
module load cudnn/9.2.0.82-12

# Activate conda environment
conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12

# Set environment variables (optional, but good practice)
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

# Python binary
PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

Run aggregation
$PYTHON_BIN aggregate_loco_results.py \
    --results_dir ${RESULTS_DIR} \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "======================================"
echo "Aggregation complete!"
echo "======================================"
echo ""
echo "Outputs saved to: ${OUTPUT_DIR}"
echo ""
echo "Files created:"
echo "  - Table 1: table1_comprehensive_results.csv (and .tex)"
echo "  - Figure 1: figure1_geogap_barchart.png"
echo "  - Figure 2: figure2_forest_plot.png"
echo "  - Summary: summary_statistics.txt"
echo ""