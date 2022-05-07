import argparse
import importlib
from typing import Union

import pandas as pd

from deepsvg import utils
from deepsvg.config import _Config
from deepsvg.svg_dataset import SVGDataset
from deepsvg.train import train
from deepsvg.utils import get_str_formatted_time

Num = Union[int, float]


def load_dataset(cfg: _Config, return_test_dataset=True):
    train_df = pd.read_csv(cfg.train_meta_filepath)
    train_dataset = SVGDataset(
        train_df,
        cfg.train_data_dir,
        cfg.model_args,
        cfg.max_num_groups,
        cfg.max_seq_len,
        cfg.max_total_len,
        nb_augmentations=cfg.nb_augmentations,
        already_preprocessed=True
    )

    val_df = pd.read_csv(cfg.val_meta_filepath)
    val_dataset = SVGDataset(
        val_df,
        cfg.val_data_dir,
        cfg.model_args,
        cfg.max_num_groups,
        cfg.max_seq_len,
        cfg.max_total_len,
        random_aug=False,
        nb_augmentations=1,
        already_preprocessed=True
    )

    if not return_test_dataset:
        print(len(train_dataset), len(val_dataset))
        return train_dataset, val_dataset

    test_df = pd.read_csv(cfg.test_meta_filepath)
    test_dataset = SVGDataset(
        test_df,
        cfg.test_data_dir,
        cfg.model_args,
        cfg.max_num_groups,
        cfg.max_seq_len,
        cfg.max_total_len,
        random_aug=False,
        nb_augmentations=1,
        already_preprocessed=True
    )
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    """
    Environment (WIP):
    torch torchvision matplotlib svgwrite cairosvg IPython tensorboardX pillow svglib tqdm jupyter scikit-image pandas networkx moviepy numba sklearn umap-learn umap-learn[plot] shapely kivy pandas sklearn wandb
    
    Evaluate on DeepSVG's icons dataset:
    python -m deepsvg.train_deepsvg --config-module svglatte.dataset.deepsvg_hierarchical_ordered_icons --num_gpus 3

    Evaluate on Argoverse:
    python -m svglatte.train_deepsvg --config-module svglatte.dataset.deepsvg_hierarchical_ordered_argoverse --num_gpus 3    
    """

    print(get_str_formatted_time())
    parser = argparse.ArgumentParser(description='DeepSVG Trainer')
    parser.add_argument("--config-module", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--eval-only", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--l1_viewbox", type=int, default=24)

    args = parser.parse_args()
    utils.set_seed(args.seed)
    print(args)

    cfg = importlib.import_module(args.config_module).Config(num_gpus=args.num_gpus)
    model_name, experiment_name = args.config_module.split(".")[-2:]

    train(cfg, model_name, experiment_name, log_dir=args.log_dir, debug=args.debug, resume=args.resume,
          eval_only=args.eval_only, eval_l1_loss=True, eval_l1_loss_viewbox=args.l1_viewbox)
