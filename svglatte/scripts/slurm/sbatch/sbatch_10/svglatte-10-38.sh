#!/bin/bash
#SBATCH --chdir /scratch/izar/rajic/svg-latte
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=90G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

#SBATCH -o ./slurm_logs/slurm-sbatch_10-38-%j.out

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
conda activate svglatte-pl
export PYTHONPATH="$PYTHONPATH:$PWD/deepsvg"

# Run
date
printf "Run configured and environment set up. Gonna run now.\n\n"
python -m svglatte.train \
 --experiment_name svg-latte-hparamsearch \
 --experiment_version 'S10.38__Residual_Lstm-06-16__e=300__bs=512__lre=8.4e-02__lrd=4.2e-04' \
 --seed 72 \
 --gpus -1 \
 --n_epochs 300 \
 --early_stopping_patience 80 \
 --check_val_every_n_epoch 10 \
 --batch_size 512 \
 --encoder_lr 0.084 \
 --decoder_lr 0.00042 \
 --encoder_weight_decay 0.0 \
 --decoder_weight_decay 0.0 \
 --encoder_type residual_lstm \
 --lstm_num_layers 6 \
 --latte_ingredients c \
 --decoder_n_filters_in_last_conv_layer 16 \
 --no_layernorm \
 --cx_loss_w 0.0 \
 --dataset argoverse \
 --argoverse_sequences_format svgtensor_data \
 --argoverse_train_sequences_path data/argoverse/train.split_1.sequences.torchsave \
 --argoverse_val_sequences_path data/argoverse/train.split_2.sequences.torchsave \
 --argoverse_test_sequences_path data/argoverse/val.sequences.torchsave \
 --argoverse_train_workers 20 \
 --argoverse_val_workers 2 \
 --argoverse_rendered_images_width 128 \
 --argoverse_rendered_images_height 128 \
 --argoverse_augment_train \
 --argoverse_zoom_preprocess_factor 0.70710678118 \
 --precision 16 \
 --gradient_clip_val 1.0 \

echo FINISHED at $(date)
