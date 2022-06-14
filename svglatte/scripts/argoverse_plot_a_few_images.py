from datetime import datetime

import torchvision
from matplotlib import pyplot as plt

from svglatte.dataset.argoverse_dataset import ArgoverseDataset


def main():
    """Plotting a few images"""
    print("load datasets")
    val_ds_1 = ArgoverseDataset(
        caching_path_prefix="/home/user72/Desktop/argoverse1/val",
        rendered_images_width=512,
        rendered_images_height=512,
        remove_redundant_features=True,
        numericalize=False,
        augment=False,
    )
    val_ds_2 = ArgoverseDataset(
        caching_path_prefix="/home/user72/Desktop/argoverse2/val",
        rendered_images_width=512,
        rendered_images_height=512,
        remove_redundant_features=True,
        numericalize=False,
        augment=True,
    )
    print("loaded")

    rendered_images = [val_ds_1[-1][1]]
    rendered_images += [val_ds_2[-1][1] for _ in range(63)]

    grid = torchvision.utils.make_grid(rendered_images)
    plt.imsave(
        f"/home/user72/Desktop/{datetime.now().strftime('%m.%d_%H.%M.%S')}.png",
        grid.permute(1, 2, 0).numpy()
    )
    plt.figure(dpi=300)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
    plt.close()

    print("done")


if __name__ == '__main__':
    main()
