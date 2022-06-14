#!/bin/bash

#SBATCH --chdir /scratch/izar/rajic/svg-latte
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH -o ./slurm_logs/slurm-sbatch_03-03-%j.out

set -e
set -o xtrace
echo PWD:$(pwd)
echo STARTING AT $(date)

# Modules
module purge
module load gcc/9.3.0-cuda
module load cuda/11.0.2

# Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate svglatte

# Run
date
printf "Run configured and environment setup. Gonna run now.\n\n"
python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_200x200_test --dataset=argoverse --gpus -1 --batch_size=64 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 8 --n_epochs 200 --argoverse_rendered_images_width 200 --argoverse_rendered_images_height 200
echo FINISHED at $(date)
