from collections import defaultdict
from datetime import datetime

import numpy as np
import torchvision
from matplotlib import pyplot as plt

from svglatte.dataset.argoverse_dataset import ArgoverseDataset


def main():
    """Visualize different viewboxes for images of different resolution."""

    datetime_str = datetime.now().strftime('%m.%d_%H.%M.%S')
    resolutions = [64, 128, 256, 512]
    viewbox_sizes = [24, 64, 128, 256]
    viewbox_sizes_str = "-".join(map(str, viewbox_sizes))
    for res in resolutions:
        rendered_images = defaultdict(list)
        for viewbox in viewbox_sizes:
            ds = ArgoverseDataset(
                caching_path_prefix="/home/user72/Desktop/argoverse1/val",
                rendered_images_width=res,
                rendered_images_height=res,
                render_on_the_fly=True,
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
                f"/home/user72/Desktop/viewbox/{datetime_str}_{id}_{res}_{viewbox_sizes_str}.png",
                grid.permute(1, 2, 0).numpy()
            )

    print("done")


if __name__ == '__main__':
    main()
