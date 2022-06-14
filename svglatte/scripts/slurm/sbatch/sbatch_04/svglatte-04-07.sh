#!/bin/bash

#SBATCH --chdir /scratch/izar/rajic/svg-latte
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=370G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --reservation=vita
#SBATCH --time=72:00:00
#SBATCH -o ./slurm_logs/slurm-sbatch_04-07-%j.out

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
python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test2 --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --lstm_num_layers 8 --n_epochs 1 --argoverse_fast_run --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm
echo FINISHED at $(date)
