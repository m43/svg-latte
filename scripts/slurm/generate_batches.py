import os

from svglatte.utils.util import ensure_dir


def fill_template(i, sbatch_id, command):
    return f"""#!/bin/bash

#SBATCH --chdir /scratch/izar/rajic/svg-latte
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G
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


sbatch01 = {
    "id": "sbatch_01",
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
}

sbatch01 = {
    "id": "sbatch_01",
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
}


sbatch02 = { # same as 01, but 2000 epochs
    "id": "sbatch_02",
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
}

sbatch02 = { # same as 01, but 2000 epochs
    "id": "sbatch_02",
    "commands": [
        "python -m svglatte.models.neural_rasterizer --experiment_name=svglatte_argoverse_200x200_test --dataset=argoverse --gpus -1 --batch_size=1024 --lr 0.0002 --weight_decay 0.0 --encoder_type fc_lstm --n_epochs 2000 --argoverse_rendered_images_width 200 --argoverse_rendered_images_height 200",
    ]
}

SBATCH = sbatch02
OUTPUT_FOLDER = f"./sbatch/{SBATCH['id']}"

if __name__ == '__main__':
    ensure_dir(OUTPUT_FOLDER)
    for i, cmd in enumerate(SBATCH["commands"]):
        i += 1  # start from 1
        script_path = os.path.join(OUTPUT_FOLDER, f"svglatte-{SBATCH['id'].split('_')[-1]}-{i:02d}.sh")
        with open(script_path, "w") as f:
            f.write(fill_template(i=i, sbatch_id=SBATCH["id"], command=cmd))
        print(f"Created script: {script_path}")
    print("Done")
