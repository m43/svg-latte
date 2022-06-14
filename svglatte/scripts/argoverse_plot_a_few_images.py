import argparse
import os
from datetime import datetime

import torchvision
from matplotlib import pyplot as plt

from svglatte.dataset.argoverse_dataset import ArgoverseDataset
from svglatte.utils.util import ensure_dir


def main(args):
    """Plotting a few images"""
    print("load datasets")
    ds_no_aug = ArgoverseDataset(
        caching_path_prefix=args.caching_path_prefix,
        rendered_images_width=512,
        rendered_images_height=512,
        remove_redundant_features=True,
        numericalize=False,
        augment=False,
    )
    ds_aug = ArgoverseDataset(
        caching_path_prefix=args.caching_path_prefix,
        rendered_images_width=512,
        rendered_images_height=512,
        remove_redundant_features=True,
        numericalize=False,
        augment=True,
    )
    print("loaded")

    # The first image in the grid will be without augmentation
    rendered_images = [ds_no_aug[-1][1]]
    # All other images will have default augmentation added
    rendered_images += [ds_aug[-1][1] for _ in range(63)]

    grid = torchvision.utils.make_grid(rendered_images)
    plt.imsave(
        os.path.join(args.plots_output_directory, f"noaug_vs_aug__{datetime.now().strftime('%m.%d_%H.%M.%S')}.png"),
        grid.permute(1, 2, 0).numpy()
    )
    plt.figure(dpi=300)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
    plt.close()

    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caching_path_prefix', type=str, default='data/argoverse/train')
    parser.add_argument('--plots_output_directory', type=str, default='logs/plot_argoverse')

    args = parser.parse_args()
    print(f"Args: {args}")

    # Create plots dir if it does not exist.
    ensure_dir(args.plots_output_directory)

    main(args)
