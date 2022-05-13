#!/bin/bash
#SBATCH --chdir /scratch/izar/rajic/svg-latte
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=370G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00

#SBATCH -o ./slurm_logs/slurm-sbatch_07-02-%j.out

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
export PYTHONPATH="$PYTHONPATH:$PWD/deepsvg"

# Run
date
printf "Run configured and environment setup. Gonna run now.\n\n"
python -m svglatte.train --experiment_name=svglatte_argoverse_128x128_rotAUG --experiment_version 'S7.02_DeepSVG_Encoder_encLR=1e-4_dimZ=1024' --gpus -1 --n_epochs 450 --early_stopping_patience 50 --batch_size=512 --encoder_lr 0.0001 --gradient_clip_val 1.0 --encoder_type deepsvg --deepsvg_encoder_hidden_size 2048 --decoder_n_filters_in_last_conv_layer 32 --no_layernorm --cx_loss_w 0.0 --dataset=argoverse --argoverse_train_workers 32 --argoverse_val_workers 15 --argoverse_rendered_images_width 128 --argoverse_rendered_images_height 128 --argoverse_render_onthefly --argoverse_augment_train --argoverse_zoom_preprocess_factor 0.70710678118
echo FINISHED at $(date)
