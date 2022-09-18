#!/bin/bash
#SBATCH --chdir /scratch/izar/rajic/svg-latte
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=180G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

#SBATCH -o ./slurm_logs/%x-%j.out

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
 --experiment_version 'S11.WD.14__latte=h__dec_filters=64__wde=0.0e+00__wdd=5.0e-06_09.15_07.48.52' \
 --seed 72 \
 --gpus -1 \
 --n_epochs 280 \
 --early_stopping_patience 80 \
 --check_val_every_n_epoch 10 \
 --batch_size 512 \
 --encoder_lr 0.00042 \
 --decoder_lr 2.1e-05 \
 --encoder_weight_decay 0.0 \
 --decoder_weight_decay 5e-06 \
 --encoder_type fc_lstm \
 --lstm_num_layers 8 \
 --latte_ingredients h \
 --decoder_n_filters_in_last_conv_layer 64 \
 --no_layernorm \
 --cx_loss_w 0.0 \
 --dataset argoverse \
 --argoverse_sequences_format svgtensor_data \
 --argoverse_train_sequences_path data/argoverse/train.split_1.sequences.torchsave \
 --argoverse_val_sequences_path data/argoverse/train.split_2.sequences.torchsave \
 --argoverse_test_sequences_path data/argoverse/val.sequences.torchsave \
 --argoverse_train_workers 1 \
 --argoverse_val_workers 1 \
 --argoverse_test_workers 18 \
 --argoverse_rendered_images_width 128 \
 --argoverse_rendered_images_height 128 \
 --argoverse_augment_train --argoverse_zoom_preprocess_factor 0.70710678118 \
 --precision 16 \
 --gradient_clip_val 1.0 \
 --do_not_add_timestamp_to_experiment_version --checkpoint_path  ./logs/svg-latte-hparamsearch/S11.WD.14__latte-h__dec_filters-64__wde-0.0e+00__wdd-5.0e-06_09.15_07.48.52_S11.WD.14__latte=h__dec_filters=64__wde=0.0e+00__wdd=5.0e-06_09.15_07.48.52/checkpoints/last.ckpt

echo FINISHED at $(date)
