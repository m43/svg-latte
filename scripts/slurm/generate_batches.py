import os

from svglatte.utils.util import ensure_dir


def fill_template(i, sbatch_id, command):
    return f"""#!/bin/bash

#SBATCH --chdir /scratch/izar/rajic/svg-latte
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=370G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
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
        "commands": [
            # experiment_version
        ]

    },
    "sbatch_07": {

    },
    "sbatch_08": {

    },
    "sbatch_09": {

    },
}

SBATCH_ID = 'sbatch_05'
OUTPUT_FOLDER = f"./sbatch/{SBATCH_ID}"

sbatch_config = sbatch_configurations[SBATCH_ID]
if __name__ == '__main__':
    ensure_dir(OUTPUT_FOLDER)
    for i, cmd in enumerate(sbatch_config["commands"]):
        i += 1  # start from 1
        script_path = os.path.join(OUTPUT_FOLDER, f"svglatte-{SBATCH_ID.split('_')[-1]}-{i:02d}.sh")
        with open(script_path, "w") as f:
            f.write(fill_template(i=i, sbatch_id=SBATCH_ID, command=cmd))
        print(f"Created script: {script_path}")
    print("Done")
