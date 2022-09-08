import argparse
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import torchvision
from matplotlib import pyplot as plt

from svglatte.dataset.argoverse_dataset import ArgoverseDataset
from svglatte.utils.util import ensure_dir


def main(args):
    """Visualize different viewboxes for images of different resolution."""
    datetime_str = datetime.now().strftime('%m.%d_%H.%M.%S')
    resolutions = [64, 128, 256, 512]
    viewbox_sizes = [24, 64, 128, 256]
    viewbox_sizes_str = "-".join(map(str, viewbox_sizes))
    for res in resolutions:
        rendered_images = defaultdict(list)
        for viewbox in viewbox_sizes:
            ds = ArgoverseDataset(
                sequences_path=args.sequences_path,
                rendered_images_width=res,
                rendered_images_height=res,
                remove_redundant_features=True,
                numericalize=False,
                augment=False,
                viewbox=viewbox,
                zoom_preprocess_factor=0.70710678118,
            )
            rendered_images["img1"].append(ds[0][1])
            rendered_images["img2"].append(ds[-1][1])

        for id, imgs in rendered_images.items():
            grid = torchvision.utils.make_grid(imgs, nrow=int(np.sqrt(len(viewbox_sizes))))
            plt.imsave(
                os.path.join(
                    args.plots_dir,
                    f"{datetime_str}__viewboxes_res={res}x{res}_vboxes={viewbox_sizes_str}.png"
                ),
                grid.permute(1, 2, 0).numpy()
            )

    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequences_path', type=str, default='data/argoverse/train.sequences.torchsave')
    parser.add_argument('--plots_dir', type=str, default='logs/plot_argoverse')

    args = parser.parse_args()
    print(f"Args: {args}")

    # Create plots dir if it does not exist.
    ensure_dir(args.plots_dir)

    main(args)
