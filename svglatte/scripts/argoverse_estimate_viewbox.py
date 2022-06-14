import torch

from svglatte.dataset.argoverse_dataset import ArgoverseDataset


def main():
    """Verify that all sequences are inside the 24 viewbox"""
    for ds_name in ["val", "test", "train"]:  # val is the smallest, train the largest subset.
        print(f"Loading subset: {ds_name}")
        ds = ArgoverseDataset(
            caching_path_prefix=f"/home/user72/Desktop/argoverse1/{ds_name}",
            rendered_images_width=64,
            rendered_images_height=64,
            remove_redundant_features=True,
            numericalize=False,
            augment=False,
        )
        print(f"Loaded the subset: {ds_name}")
        print(f"Number of datapoints: {len(ds._sequences)}")

        def set_unused_to_some_constant(seq, const=5):
            # I put five because I expect that my data has a veiwbox of 24, so the xy coordinates
            # are all in the box (0,0)-(24,24).
            return (seq != -1) * seq + (seq == -1) * const

        print(torch.cat([set_unused_to_some_constant(seq).max(0).values[None, :] for seq in ds._sequences]).max(0))
        print(torch.cat([set_unused_to_some_constant(seq).min(0).values[None, :] for seq in ds._sequences]).min(0))

        print(f"Done with {ds}")
        print()
        print()


if __name__ == '__main__':
    main()
