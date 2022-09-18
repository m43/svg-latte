import os
import pathlib
import random

DEBUG_HEADER = """#SBATCH --chdir /scratch/izar/rajic/svg-latte
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=50G
#SBATCH --partition=debug
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00
"""

PRODUCTION_HEADER_1_GPU = """#SBATCH --chdir /scratch/izar/rajic/svg-latte
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=90G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
"""

PRODUCTION_HEADER_2_GPUS = """#SBATCH --chdir /scratch/izar/rajic/svg-latte
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
"""

PRODUCTION_HEADER_2_GPUS_W_RAM = """#SBATCH --chdir /scratch/izar/rajic/svg-latte
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=370G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
"""


def fill_template(command, header):
    return f"""#!/bin/bash
{header}
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
printf "Run configured and environment set up. Gonna run now.\\n\\n"
{command}
echo FINISHED at $(date)
"""


sbatch_configurations = {
    "sbatch_01": {
        "commands": [
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test_batch --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --n_epochs 2",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test_batch --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --latte_ingredients h --n_epochs 2",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test_batch --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --latte_ingredients c --n_epochs 2",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test_batch --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --lstm_num_layers 8 --n_epochs 2",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test_batch --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --lstm_bidirectional --n_epochs 2",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test_batch --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --no_sequence_packing --n_epochs 2",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test_batch --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm_original --n_epochs 2",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test_batch --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type dvf_lstm --no_sequence_packing --n_epochs 2",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test_batch --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type lstm --n_epochs 2",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test_batch --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type lstm --lstm_bidirectional --n_epochs 2",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test_batch --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --n_epochs 2",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test_batch --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_bidirectional --n_epochs 2",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test_batch --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type lstm+mha --lstm_bidirectional --n_epochs 2",
        ]
    },
    "sbatch_02": {  # same as 01, but 2000 epochs
        "commands": [
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --n_epochs 2000",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --latte_ingredients h --n_epochs 2000",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --latte_ingredients c --n_epochs 2000",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --lstm_num_layers 8 --n_epochs 2000",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --lstm_bidirectional --n_epochs 2000",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --no_sequence_packing --n_epochs 2000",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm_original --n_epochs 2000",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type dvf_lstm --no_sequence_packing --n_epochs 2000",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type lstm --n_epochs 2000",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type lstm --lstm_bidirectional --n_epochs 2000",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --n_epochs 2000",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_bidirectional --n_epochs 2000",
        ]
    },
    "sbatch_03": {
        "commands": [
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_200x200_test --dataset=argoverse --gpus -1 --batch_size=64 --lr 0.0002 --weight_decay 0.0 --encoder_type lstm --no_layernorm --n_epochs 200 --argoverse_rendered_images_width 200 --argoverse_rendered_images_height 200",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_200x200_test --dataset=argoverse --gpus -1 --batch_size=64 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --n_epochs 200 --argoverse_rendered_images_width 200 --argoverse_rendered_images_height 200",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_200x200_test --dataset=argoverse --gpus -1 --batch_size=64 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 8 --n_epochs 200 --argoverse_rendered_images_width 200 --argoverse_rendered_images_height 200",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_200x200_test --dataset=argoverse --gpus -1 --batch_size=64 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 8 --no_layernorm --n_epochs 200 --argoverse_rendered_images_width 200 --argoverse_rendered_images_height 200",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_200x200_test --dataset=argoverse --gpus -1 --batch_size=64 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 12 --n_epochs 200 --argoverse_rendered_images_width 200 --argoverse_rendered_images_height 200",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_200x200_test --dataset=argoverse --gpus -1 --batch_size=64 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --n_epochs 200 --argoverse_rendered_images_width 200 --argoverse_rendered_images_height 200",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_200x200_test --dataset=argoverse --gpus -1 --batch_size=64 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --lstm_num_layers 8 --n_epochs 200 --argoverse_rendered_images_width 200 --argoverse_rendered_images_height 200",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_200x200_test --dataset=argoverse --gpus -1 --batch_size=64 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --lstm_num_layers 8 --latte_ingredients c --n_epochs 200 --argoverse_rendered_images_width 200 --argoverse_rendered_images_height 200",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_200x200_test --dataset=argoverse --gpus -1 --batch_size=64 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --lstm_num_layers 8 --no_layernorm --n_epochs 200 --argoverse_rendered_images_width 200 --argoverse_rendered_images_height 200",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_200x200_test --dataset=argoverse --gpus -1 --batch_size=64 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --lstm_num_layers 12 --n_epochs 200 --argoverse_rendered_images_width 200 --argoverse_rendered_images_height 200",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_200x200_test --dataset=argoverse --gpus -1 --batch_size=64 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --lstm_num_layers 12 --no_layernorm --n_epochs 200 --argoverse_rendered_images_width 200 --argoverse_rendered_images_height 200",
        ]
    },
    "sbatch_04": {  # Test setup
        "commands": [
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test2 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 1 --argoverse_fast_run --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --no_layernorm",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test2 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 1 --argoverse_fast_run --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test2 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 1 --argoverse_fast_run --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test2 --dataset=argoverse --argoverse_train_workers 4 --argoverse_val_workers 4  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 1 --argoverse_fast_run --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.1 --argoverse_augment_train --no_layernorm",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test2 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 1 --argoverse_fast_run --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm --argoverse_augment_scale_min 0.9 --argoverse_augment_scale_max 1.5",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test2 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 1 --argoverse_fast_run --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm --argoverse_numericalize",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test2 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --lstm_num_layers 8 --n_epochs 1 --argoverse_fast_run --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm",
        ]
    },
    "sbatch_05": {  # Same as 04, but not a test
        "commands": [
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 200 --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --no_layernorm --experiment_version '5s01 noAUG Residual_noLN_noCX 4,512,hc'",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 200 --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --experiment_version '5s02 AUG Residual_noCX 4,512,hc'",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 200 --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm --experiment_version '5s03 AUG Residual_noLN_noCX 4,512,hc'",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 4 --argoverse_val_workers 4  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 200 --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.1 --argoverse_augment_train --no_layernorm --experiment_version '5s04 AUG Residual_noLN 4,512,hc'",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 200 --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm --argoverse_augment_scale_mn 0.9 --argoverse_augment_scale_max 1.5 --experiment_version '5s05 AUG_ZOOM Residual_noLN_noCX 4,512,hc'",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 200 --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm --argoverse_numericalize --experiment_version '5s06 AUG Residual_noLN_noCX_nize 4,512,hc'",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --lstm_num_layers 8 --n_epochs 200 --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm --experiment_version '5s07 AUG FCLSTM_noLN_noCX 8,512,hc'",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 200 --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm --argoverse_augment_scale_mn 1.5 --argoverse_augment_scale_max 2.5 --experiment_version '5s08 AUG_ZOOM2 Residual_noLN_noCX 4,512,hc'",
        ]
    },
    "sbatch_06": {
        "debug": False,
        "commands": [
            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S6.01_FC.4c_rotAUG_noLN_noCX'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --lr 0.00042"
            " --weight_decay 0.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 16"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S6.02_FC.4c_noAUG_noLN_noCX'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --lr 0.00042"
            " --weight_decay 0.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 16"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S6.03_FC.4c_rotAUG_noLN_noCX_WD=0.0001'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --lr 0.00042"
            " --weight_decay 0.0001"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 16"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S6.04_FC.4c_rotAUG_noLN_noCX_NGF=32'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --lr 0.00042"
            " --weight_decay 0.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 32"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S6.05_FC.4c_rotAUG_LN_noCX'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --lr 0.00042"
            " --weight_decay 0.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 16"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S6.06_FC.4c_rotAUG_noLN_noCX_GC=1.0'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --lr 0.00042"
            " --weight_decay 0.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 16"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_zoom_preprocess_factor 0.70710678118"
            " --argoverse_augment_train"
            " --gradient_clip_val 1.0",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S6.07_FC.4c_rotAUG_noLN_CX'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --lr 0.00042"
            " --weight_decay 0.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 16"
            " --no_layernorm"
            " --cx_loss_w 0.1"
            " --dataset=argoverse"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_zoom_preprocess_factor 0.70710678118"
            " --argoverse_augment_train",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S6.08_FC.4c_rotAUG_noLN_noCX_WD=0.000005'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --lr 0.00042"
            " --weight_decay 0.000005"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 16"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S6.09_FC.4c_rotAUG_noLN_noCX_viewbox=128'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --lr 0.00042"
            " --weight_decay 0.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 16"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118"
            " --argoverse_viewbox 128",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S6.10_FC.4c_rotAUG_noLN_noCX_BN'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --lr 0.00042"
            " --weight_decay 0.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 16"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118"
            " --decoder_norm_layer_name batchnorm",
        ]

    },
    "sbatch_07": {
        "debug": False,
        "commands": [
            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S7.01_DeepSVG_Encoder'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --gradient_clip_val 1.0"
            " --encoder_type deepsvg"
            " --decoder_n_filters_in_last_conv_layer 32"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S7.02_DeepSVG_Encoder_encLR=1e-4_dimZ=1024'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --encoder_lr 0.0001"
            " --gradient_clip_val 1.0"
            " --encoder_type deepsvg"
            " --deepsvg_encoder_hidden_size 2048"
            " --decoder_n_filters_in_last_conv_layer 32"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_train_workers 32"
            " --argoverse_val_workers 15"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

        ]

    },
    "sbatch_08": {  # Svg-latte with new and corrected Argoverse dataset, the one we will use with SVG-Net
        "debug": False,
        "commands": [
            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S8.01_ARGO2_FC.4c_rotAUG_noLN_noCX_NGF=32_GC=1.0'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --encoder_lr 0.00042"
            " --decoder_lr 0.00042"
            " --encoder_weight_decay 0.0"
            " --decoder_weight_decay 0.0"
            " --gradient_clip_val 1.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 32"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_cached_sequences_format svgtensor_data"
            " --argoverse_data_root /scratch/izar/rajic/svgnet-hossein/cache/default/"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S8.02_ARGO2_FC.4c_rotAUG_noLN_noCX_NGF=32_GC=1.0'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --encoder_lr 0.00042"
            " --decoder_lr 0.00042"
            " --encoder_weight_decay 0.0"
            " --decoder_weight_decay 0.0"
            " --gradient_clip_val 1.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 32"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_cached_sequences_format svgtensor_data"
            " --argoverse_data_root /scratch/izar/rajic/svgnet-hossein/cache/default/"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S8.03_ARGO2_FC.4c_noAUG_noLN_noCX_NGF=32_GC=1.0'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --encoder_lr 0.00042"
            " --decoder_lr 0.00042"
            " --encoder_weight_decay 0.0"
            " --decoder_weight_decay 0.0"
            " --gradient_clip_val 1.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 32"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_cached_sequences_format svgtensor_data"
            " --argoverse_data_root /scratch/izar/rajic/svgnet-hossein/cache/default/"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S8.05_ARGO2_Res.4hc_noAUG_noLN_noCX_NGF=32_GC=1.0_encoderLR=decoderLR=0.0001'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --encoder_lr 0.0001"
            " --decoder_lr 0.0001"
            " --encoder_weight_decay 0.0"
            " --decoder_weight_decay 0.0"
            " --gradient_clip_val 1.0"
            " --encoder_type residual_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients hc"
            " --decoder_n_filters_in_last_conv_layer 32"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_cached_sequences_format svgtensor_data"
            " --argoverse_data_root /scratch/izar/rajic/svgnet-hossein/cache/default/"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_zoom_preprocess_factor 0.70710678118",
        ]

    },
    "sbatch_09": {
        "debug": False,
        "commands": [
            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S9.01_ARGO4_FC.4c_rotAUG_noLN_noCX_NGF=16_GC=None'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --encoder_lr 0.00042"
            " --decoder_lr 0.00042"
            " --encoder_weight_decay 0.0"
            " --decoder_weight_decay 0.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 16"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_cached_sequences_format svgtensor_data"
            " --argoverse_data_root /scratch/izar/rajic/svgnet-hossein/cache/argo_4/"
            " --argoverse_train_workers 30"
            " --argoverse_val_workers 15"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S9.02_ARGO4_FC.4c_rotAUG_noLN_noCX_NGF=32_GC=1.0'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --encoder_lr 0.00042"
            " --decoder_lr 0.00042"
            " --encoder_weight_decay 0.0"
            " --decoder_weight_decay 0.0"
            " --gradient_clip_val 1.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 32"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_cached_sequences_format svgtensor_data"
            " --argoverse_data_root /scratch/izar/rajic/svgnet-hossein/cache/argo_4/"
            " --argoverse_train_workers 30"
            " --argoverse_val_workers 15"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S9.03_ARGO4_FC.4c_noAUG_noLN_noCX_NGF=32_GC=1.0'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --encoder_lr 0.00042"
            " --decoder_lr 0.00042"
            " --encoder_weight_decay 0.0"
            " --decoder_weight_decay 0.0"
            " --gradient_clip_val 1.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 32"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_cached_sequences_format svgtensor_data"
            " --argoverse_data_root /scratch/izar/rajic/svgnet-hossein/cache/argo_4/"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S9.04_ARGO4_Res.4hc_rotAUG_noLN_noCX_NGF=32_GC=1.0_encoderLR=decoderLR=0.0001'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --encoder_lr 0.0001"
            " --decoder_lr 0.0001"
            " --encoder_weight_decay 0.0"
            " --decoder_weight_decay 0.0"
            " --gradient_clip_val 1.0"
            " --encoder_type residual_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients hc"
            " --decoder_n_filters_in_last_conv_layer 32"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_cached_sequences_format svgtensor_data"
            " --argoverse_data_root /scratch/izar/rajic/svgnet-hossein/cache/argo_4/"
            " --argoverse_train_workers 40"
            " --argoverse_val_workers 10"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S9.05_256x256_vbox=24__ARGO4_FC.4c_rotAUG_noLN_noCX_NGF=16_GC=None'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --encoder_lr 0.00042"
            " --decoder_lr 0.00042"
            " --encoder_weight_decay 0.0"
            " --decoder_weight_decay 0.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 16"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_cached_sequences_format svgtensor_data"
            " --argoverse_data_root /scratch/izar/rajic/svgnet-hossein/cache/argo_4/"
            " --argoverse_train_workers 30"
            " --argoverse_val_workers 15"
            " --argoverse_rendered_images_width 256"
            " --argoverse_rendered_images_height 256"
            " --argoverse_viewbox 24"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S9.06_256x256_vbox=64__ARGO4_FC.4c_rotAUG_noLN_noCX_NGF=16_GC=None'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=256"
            " --encoder_lr 0.00042"
            " --decoder_lr 0.00042"
            " --encoder_weight_decay 0.0"
            " --decoder_weight_decay 0.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 16"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_cached_sequences_format svgtensor_data"
            " --argoverse_data_root /scratch/izar/rajic/svgnet-hossein/cache/argo_4/"
            " --argoverse_train_workers 30"
            " --argoverse_val_workers 15"
            " --argoverse_rendered_images_width 256"
            " --argoverse_rendered_images_height 256"
            " --argoverse_viewbox 64"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S9.07_128x128_vbox=40__ARGO4_FC.4c_rotAUG_noLN_noCX_NGF=16_GC=None'"
            " --gpus -1"
            " --n_epochs 450"
            " --early_stopping_patience 50"
            " --batch_size=512"
            " --encoder_lr 0.00042"
            " --decoder_lr 0.00042"
            " --encoder_weight_decay 0.0"
            " --decoder_weight_decay 0.0"
            " --encoder_type fc_lstm"
            " --lstm_num_layers 4"
            " --latte_ingredients c"
            " --decoder_n_filters_in_last_conv_layer 16"
            " --no_layernorm"
            " --cx_loss_w 0.0"
            " --dataset=argoverse"
            " --argoverse_cached_sequences_format svgtensor_data"
            " --argoverse_data_root /scratch/izar/rajic/svgnet-hossein/cache/argo_4/"
            " --argoverse_train_workers 30"
            " --argoverse_val_workers 15"
            " --argoverse_rendered_images_width 128"
            " --argoverse_rendered_images_height 128"
            " --argoverse_viewbox 40"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",
        ]

    },
    "sbatch_10": {
        "debug": False,
        "commands": [x.format(i=i + 1) for i, x in enumerate([
            f"python -m svglatte.train \\\n"
            f" --experiment_name svg-latte-hparamsearch \\\n"
            f" --experiment_version "
            f"'S10.{{i:02d}}__{enc_type.title()}-{lstm_layers:02d}-{dec_filters:02d}__e={e:03d}__bs={bs:03d}__lre={lr_e:.1e}__lrd={lr_d:.1e}' \\\n"
            f" --seed {seed} \\\n"
            f" --gpus -1 \\\n"
            f" --n_epochs {e} \\\n"
            f" --early_stopping_patience 80 \\\n"
            f" --check_val_every_n_epoch {val_every} \\\n"
            f" --batch_size {bs} \\\n"
            f" --encoder_lr {lr_e} \\\n"
            f" --decoder_lr {lr_d} \\\n"
            f" --encoder_weight_decay {wd_e} \\\n"
            f" --decoder_weight_decay {wd_d} \\\n"
            f" --encoder_type {enc_type} \\\n"
            f" --lstm_num_layers {lstm_layers} \\\n"
            f" --latte_ingredients {latte} \\\n"
            f" --decoder_n_filters_in_last_conv_layer {dec_filters} \\\n"
            f" --no_layernorm \\\n"
            f" --cx_loss_w {cx_loss} \\\n"
            f" --dataset argoverse \\\n"
            f" --argoverse_sequences_format svgtensor_data \\\n"
            f" --argoverse_train_sequences_path data/argoverse/train.split_1.sequences.torchsave \\\n"
            f" --argoverse_val_sequences_path data/argoverse/train.split_2.sequences.torchsave \\\n"
            f" --argoverse_test_sequences_path data/argoverse/val.sequences.torchsave \\\n"
            f" --argoverse_train_workers 20 \\\n"
            f" --argoverse_val_workers 2 \\\n"
            f" --argoverse_rendered_images_width 128 \\\n"
            f" --argoverse_rendered_images_height 128 \\\n"
            f" --argoverse_augment_train \\\n"
            f" --argoverse_zoom_preprocess_factor 0.70710678118 \\\n"
            f" --precision 16 \\\n"
            f" --gradient_clip_val 1.0 \\\n"
            # Varying:
            for (enc_type, lstm_layers) in [("fc_lstm", 8), ("fc_lstm", 4), ("residual_lstm", 6)]
            for e, bs, val_every in [(300, 512, 10), (100, 32, 5)]
            for lr_e in [0.084, 0.0084, 0.00042]
            for lr_d in [0.0084, 0.00042, 0.000021]
            # Fixed:
            for seed in [72]
            for wd_e, wd_d in [(0.0, 0.0)]
            for latte in ["c"]
            for dec_filters in [16]
            for cx_loss in [0.0]
        ])]
    },
    "sbatch_11w": {
        "debug": False,
        "commands": [x.format(i=i + 1) for i, x in enumerate([
            f"python -m svglatte.train \\\n"
            f" --experiment_name svg-latte-hparamsearch \\\n"
            f" --experiment_version "
            f"'S11.WD.{{i:02d}}__latte={latte}__dec_filters={dec_filters}__wde={wd_e:.1e}__wdd={wd_d:.1e}' \\\n"
            f" --seed {seed} \\\n"
            f" --gpus -1 \\\n"
            f" --n_epochs {e} \\\n"
            f" --early_stopping_patience 80 \\\n"
            f" --check_val_every_n_epoch {val_every} \\\n"
            f" --batch_size {bs} \\\n"
            f" --encoder_lr {lr_e} \\\n"
            f" --decoder_lr {lr_d} \\\n"
            f" --encoder_weight_decay {wd_e} \\\n"
            f" --decoder_weight_decay {wd_d} \\\n"
            f" --encoder_type {enc_type} \\\n"
            f" --lstm_num_layers {lstm_layers} \\\n"
            f" --latte_ingredients {latte} \\\n"
            f" --decoder_n_filters_in_last_conv_layer {dec_filters} \\\n"
            f" --no_layernorm \\\n"
            f" --cx_loss_w {cx_loss} \\\n"
            f" --dataset argoverse \\\n"
            f" --argoverse_sequences_format svgtensor_data \\\n"
            f" --argoverse_train_sequences_path data/argoverse/train.split_1.sequences.torchsave \\\n"
            f" --argoverse_val_sequences_path data/argoverse/train.split_2.sequences.torchsave \\\n"
            f" --argoverse_test_sequences_path data/argoverse/val.sequences.torchsave \\\n"
            f" --argoverse_train_workers {40 if dec_filters == 64 else 20} \\\n"
            f" --argoverse_val_workers {4 if dec_filters == 64 else 3} \\\n"
            f" --argoverse_rendered_images_width 128 \\\n"
            f" --argoverse_rendered_images_height 128 \\\n"
            f" {augment_train_flag} --argoverse_zoom_preprocess_factor 0.70710678118 \\\n"
            f" --precision 16 \\\n"
            f" --gradient_clip_val 1.0 \\\n"
            # Varying:
            for wd_e in [0.0, 0.000005]
            for wd_d in [0.0, 0.00001, 0.000005]
            for latte in ["h", "hc", "o"]
            for dec_filters in [16, 64]
            # Fixed:
            for seed in [72]
            for enc_type in ["fc_lstm"]
            for lstm_layers in [8]
            for e in [300]
            for bs in [512]
            for augment_train_flag in ["--argoverse_augment_train"]  # [""]
            for val_every in [10]
            for lr_e in [0.00042]
            for lr_d in [0.000021]
            for cx_loss in [0.0]
        ])]
    },
    "sbatch_12": {
        "debug": False,
        "commands": [x.format(i=i + 1) for i, x in enumerate([
            f"python -m svglatte.train \\\n"
            f" --experiment_name svg-latte-hparamsearch \\\n"
            f" --experiment_version "
            f"'S12.{{i:02d}}__BestHparams__Seed={seed}' \\\n"
            f" --seed {seed} \\\n"
            f" --gpus -1 \\\n"
            f" --n_epochs {e} \\\n"
            f" --early_stopping_patience 80 \\\n"
            f" --check_val_every_n_epoch {val_every} \\\n"
            f" --batch_size {bs} \\\n"
            f" --encoder_lr {lr_e} \\\n"
            f" --decoder_lr {lr_d} \\\n"
            f" --encoder_weight_decay {wd_e} \\\n"
            f" --decoder_weight_decay {wd_d} \\\n"
            f" --encoder_type {enc_type} \\\n"
            f" --lstm_num_layers {lstm_layers} \\\n"
            f" --latte_ingredients {latte} \\\n"
            f" --decoder_n_filters_in_last_conv_layer {dec_filters} \\\n"
            f" --no_layernorm \\\n"
            f" --cx_loss_w {cx_loss} \\\n"
            f" --dataset argoverse \\\n"
            f" --argoverse_sequences_format svgtensor_data \\\n"
            f" --argoverse_train_sequences_path data/argoverse/train.sequences.torchsave \\\n"
            f" --argoverse_val_sequences_path data/argoverse/val.sequences.torchsave \\\n"
            f" --argoverse_test_sequences_path data/argoverse/test.sequences.torchsave \\\n"
            f" --argoverse_train_workers 40 \\\n"
            f" --argoverse_val_workers 3 \\\n"
            f" --argoverse_rendered_images_width 128 \\\n"
            f" --argoverse_rendered_images_height 128 \\\n"
            f" {augment_train_flag} --argoverse_zoom_preprocess_factor 0.70710678118 \\\n"
            f" --precision 16 \\\n"
            f" --gradient_clip_val 1.0 \\\n"
            # Varying:
            for seed in [72, 9, 27, 32, 36, 54, 64, 180]
            # Fixed:
            for enc_type in ["fc_lstm"]
            for lstm_layers in [8]
            for e in [600]
            for bs in [512]
            for augment_train_flag in ["--argoverse_augment_train"]  # [""]
            for val_every in [20]
            for lr_e in [0.00042]
            for lr_d in [0.000021]
            for wd_e in [0.0]
            for wd_d in [0.0]
            for latte in ["c"]
            for dec_filters in [16]
            for cx_loss in [0.0]
        ])]
    },
    "sbatch_13": {
        "debug": False,
        "commands": [x.format(i=i + 1) for i, x in enumerate([
            f"python -m svglatte.train \\\n"
            f" --experiment_name svg-latte-hparamsearch \\\n"
            f" --experiment_version "
            f"'S13.{{i:02d}}__BestHparams__Viewbox={argoverse_viewbox}__Seed={seed}' \\\n"
            f" --seed {seed} \\\n"
            f" --gpus -1 \\\n"
            f" --n_epochs {e} \\\n"
            f" --early_stopping_patience 80 \\\n"
            f" --check_val_every_n_epoch {val_every} \\\n"
            f" --batch_size {bs} \\\n"
            f" --encoder_lr {lr_e} \\\n"
            f" --decoder_lr {lr_d} \\\n"
            f" --encoder_weight_decay {wd_e} \\\n"
            f" --decoder_weight_decay {wd_d} \\\n"
            f" --encoder_type {enc_type} \\\n"
            f" --lstm_num_layers {lstm_layers} \\\n"
            f" --latte_ingredients {latte} \\\n"
            f" --decoder_n_filters_in_last_conv_layer {dec_filters} \\\n"
            f" --no_layernorm \\\n"
            f" --cx_loss_w {cx_loss} \\\n"
            f" --dataset argoverse \\\n"
            f" --argoverse_sequences_format svgtensor_data \\\n"
            f" --argoverse_train_sequences_path data/argoverse/train.sequences.torchsave \\\n"
            f" --argoverse_val_sequences_path data/argoverse/val.sequences.torchsave \\\n"
            f" --argoverse_test_sequences_path data/argoverse/test.sequences.torchsave \\\n"
            f" --argoverse_train_workers 40 \\\n"
            f" --argoverse_val_workers 3 \\\n"
            f" --argoverse_rendered_images_width 128 \\\n"
            f" --argoverse_rendered_images_height 128 \\\n"
            f" {augment_train_flag} --argoverse_zoom_preprocess_factor 0.70710678118 \\\n"
            f" --precision 16 \\\n"
            f" --gradient_clip_val 1.0 \\\n"
            f" --argoverse_viewbox {argoverse_viewbox} \\\n"
            # Special:
            for argoverse_viewbox in [64]
            # Fixed:
            for seed in [72]
            for enc_type in ["fc_lstm"]
            for lstm_layers in [8]
            for e in [600]
            for bs in [512]
            for augment_train_flag in ["--argoverse_augment_train"]  # [""]
            for val_every in [20]
            for lr_e in [0.00042]
            for lr_d in [0.000021]
            for wd_e in [0.0]
            for wd_d in [0.0]
            for latte in ["c"]
            for dec_filters in [16]
            for cx_loss in [0.0]
        ])]
    },
    "sbatch_14": {
        "debug": False,
        "commands": [x.format(i=i + 1) for i, x in enumerate([
            f"python -m svglatte.train \\\n"
            f" --experiment_name svg-latte-hparamsearch \\\n"
            f" --experiment_version "
            f"'S14.{{i:02d}}__BestHparams__CX-Loss={cx_loss:.1e}__Seed={seed}' \\\n"
            f" --seed {seed} \\\n"
            f" --gpus -1 \\\n"
            f" --n_epochs {e} \\\n"
            f" --early_stopping_patience 80 \\\n"
            f" --check_val_every_n_epoch {val_every} \\\n"
            f" --batch_size {bs} \\\n"
            f" --encoder_lr {lr_e} \\\n"
            f" --decoder_lr {lr_d} \\\n"
            f" --encoder_weight_decay {wd_e} \\\n"
            f" --decoder_weight_decay {wd_d} \\\n"
            f" --encoder_type {enc_type} \\\n"
            f" --lstm_num_layers {lstm_layers} \\\n"
            f" --latte_ingredients {latte} \\\n"
            f" --decoder_n_filters_in_last_conv_layer {dec_filters} \\\n"
            f" --no_layernorm \\\n"
            f" --cx_loss_w {cx_loss} \\\n"
            f" --dataset argoverse \\\n"
            f" --argoverse_sequences_format svgtensor_data \\\n"
            f" --argoverse_train_sequences_path data/argoverse/train.sequences.torchsave \\\n"
            f" --argoverse_val_sequences_path data/argoverse/val.sequences.torchsave \\\n"
            f" --argoverse_test_sequences_path data/argoverse/test.sequences.torchsave \\\n"
            f" --argoverse_train_workers 40 \\\n"
            f" --argoverse_val_workers 3 \\\n"
            f" --argoverse_rendered_images_width 128 \\\n"
            f" --argoverse_rendered_images_height 128 \\\n"
            f" {augment_train_flag} --argoverse_zoom_preprocess_factor 0.70710678118 \\\n"
            f" --precision 16 \\\n"
            f" --gradient_clip_val 1.0 \\\n"
            # Special:
            for cx_loss in [0.1]
            for bs in [256]
            # Fixed:
            for seed in [72]
            for enc_type in ["fc_lstm"]
            for lstm_layers in [8]
            for e in [600]
            for augment_train_flag in ["--argoverse_augment_train"]  # [""]
            for val_every in [20]
            for lr_e in [0.00042]
            for lr_d in [0.000021]
            for wd_e in [0.0]
            for wd_d in [0.0]
            for latte in ["c"]
            for dec_filters in [16]
        ])]
    },
}

random.seed(72)
random.shuffle(sbatch_configurations["sbatch_11w"]["commands"])
n = len(sbatch_configurations["sbatch_11w"]["commands"])
sbatch_configurations["sbatch_11w"]["commands"] = sbatch_configurations["sbatch_11w"]["commands"][:int(n * 0.8)]

SBATCH_IDS = [
    # 'sbatch_01',
    # 'sbatch_02',
    # 'sbatch_03',
    # 'sbatch_04',
    # 'sbatch_05',
    # 'sbatch_06',
    # 'sbatch_07',
    # 'sbatch_08',
    # 'sbatch_09',
    # 'sbatch_10',
    # 'sbatch_11',
    # 'sbatch_11w',
    'sbatch_12',
    'sbatch_13',
    'sbatch_14',
]
for SBATCH_ID in SBATCH_IDS:
    OUTPUT_FOLDER = f"./sbatch/{SBATCH_ID}"

    sbatch_config = sbatch_configurations[SBATCH_ID]
    if __name__ == '__main__':
        dirname = pathlib.Path(OUTPUT_FOLDER)
        if not dirname.is_dir():
            dirname.mkdir(parents=True, exist_ok=False)

        for i, cmd in enumerate(sbatch_config["commands"]):
            i += 1  # start from 1
            sbatch_id = SBATCH_ID if not "_latte=o_" in cmd else SBATCH_ID + "o"
            script_path = os.path.join(OUTPUT_FOLDER, f"svglatte-{sbatch_id.split('_')[-1]}-{i:02d}.sh")
            with open(script_path, "w") as f:
                if sbatch_config.get("debug", False):
                    header = DEBUG_HEADER
                else:
                    if "dec_filters=64" in cmd:
                        header = PRODUCTION_HEADER_2_GPUS
                    elif "BestHparams" in cmd:
                        header = PRODUCTION_HEADER_2_GPUS_W_RAM
                    else:
                        header = PRODUCTION_HEADER_1_GPU
                f.write(fill_template(command=cmd, header=header))
            print(f"Created script: {script_path}")
        print("Done")
