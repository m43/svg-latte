import os
import pathlib

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

PRODUCTION_HEADER = """#SBATCH --chdir /scratch/izar/rajic/svg-latte
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=370G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
"""


def fill_template(i, sbatch_id, command, debug):
    return f"""#!/bin/bash
{DEBUG_HEADER if debug else PRODUCTION_HEADER}
#SBATCH -o ./slurm_logs/slurm-{sbatch_id}-{i:02d}-%j.out

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
printf "Run configured and environment setup. Gonna run now.\\n\\n"
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
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test2 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 1 --argoverse_fast_run --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --no_layernorm",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test2 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 1 --argoverse_fast_run --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test2 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 1 --argoverse_fast_run --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test2 --dataset=argoverse --argoverse_train_workers 4 --argoverse_val_workers 4  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 1 --argoverse_fast_run --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.1 --argoverse_augment_train --no_layernorm",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test2 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 1 --argoverse_fast_run --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm --argoverse_augment_scale_min 0.9 --argoverse_augment_scale_max 1.5",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test2 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 1 --argoverse_fast_run --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm --argoverse_numericalize",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64_test2 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --lstm_num_layers 8 --n_epochs 1 --argoverse_fast_run --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm",
        ]
    },
    "sbatch_05": {  # Same as 04, but not a test
        "commands": [
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 200 --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --no_layernorm --experiment_version '5s01 noAUG Residual_noLN_noCX 4,512,hc'",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 200 --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --experiment_version '5s02 AUG Residual_noCX 4,512,hc'",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 200 --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm --experiment_version '5s03 AUG Residual_noLN_noCX 4,512,hc'",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 4 --argoverse_val_workers 4  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 200 --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.1 --argoverse_augment_train --no_layernorm --experiment_version '5s04 AUG Residual_noLN 4,512,hc'",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 200 --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm --argoverse_augment_scale_mn 0.9 --argoverse_augment_scale_max 1.5 --experiment_version '5s05 AUG_ZOOM Residual_noLN_noCX 4,512,hc'",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 200 --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm --argoverse_numericalize --experiment_version '5s06 AUG Residual_noLN_noCX_nize 4,512,hc'",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --lstm_num_layers 8 --n_epochs 200 --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm --experiment_version '5s07 AUG FCLSTM_noLN_noCX 8,512,hc'",
            "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_64x64 --dataset=argoverse --argoverse_train_workers 40 --argoverse_val_workers 20  --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type residual_lstm --lstm_num_layers 4 --n_epochs 200 --argoverse_render_onthefly --argoverse_rendered_images_width 64 --argoverse_rendered_images_height 64 --cx_loss_w 0.0 --argoverse_augment_train --no_layernorm --argoverse_augment_scale_mn 1.5 --argoverse_augment_scale_max 2.5 --experiment_version '5s08 AUG_ZOOM2 Residual_noLN_noCX 4,512,hc'",
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S9.03_ARGO2_FC.4c_noAUG_noLN_noCX_NGF=32_GC=1.0'"
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
            " --argoverse_render_onthefly"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S9.04_ARGO2_Res.4hc_rotAUG_noLN_noCX_NGF=32_GC=1.0_encoderLR=decoderLR=0.0001'"
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
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",

            "python -m svglatte.train"
            " --experiment_name=svglatte_argoverse_128x128_rotAUG"
            " --experiment_version 'S9.06_256x256_vbox=64__ARGO4_FC.4c_rotAUG_noLN_noCX_NGF=16_GC=None'"
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
            " --argoverse_viewbox 64"
            " --argoverse_render_onthefly"
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
            " --argoverse_render_onthefly"
            " --argoverse_augment_train"
            " --argoverse_zoom_preprocess_factor 0.70710678118",
        ]

    },
}

SBATCH_ID = 'sbatch_09'
OUTPUT_FOLDER = f"./sbatch/{SBATCH_ID}"

sbatch_config = sbatch_configurations[SBATCH_ID]
if __name__ == '__main__':
    dirname = pathlib.Path(OUTPUT_FOLDER)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

    for i, cmd in enumerate(sbatch_config["commands"]):
        i += 1  # start from 1
        script_path = os.path.join(OUTPUT_FOLDER, f"svglatte-{SBATCH_ID.split('_')[-1]}-{i:02d}.sh")
        with open(script_path, "w") as f:
            f.write(fill_template(i=i, sbatch_id=SBATCH_ID, command=cmd, debug=sbatch_config["debug"]))
        print(f"Created script: {script_path}")
    print("Done")
