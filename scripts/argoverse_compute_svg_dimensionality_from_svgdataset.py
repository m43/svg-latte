import pandas as pd


def main():
    """
    Count the real maximum values for max_num_groups, max_seq_len, and max_total_len.
    They might be lower than the numbers used when creating Argoverse in SVG-Net, i.e.
        self.max_num_groups = 120
        self.max_seq_len = 200
        self.max_total_len = 2000
    If the numbers are really lower, then it will be easier to evaluate them with DeepSVG.
    """
    for subset, csv_path in [
        ("train", "data/svgnet-hossein-argoverse-4/svgdataset/train/svg_meta.csv"),
        ("val", "data/svgnet-hossein-argoverse-4/svgdataset/val/svg_meta.csv"),
        ("test", "data/svgnet-hossein-argoverse-4/svgdataset/test/svg_meta.csv"),
        # ("val", "data/argoverse10/val/svg_meta.csv"),
    ]:
        print(f"Subset: {subset}")
        df = pd.read_csv(csv_path)
        print(f"Dataset length: {len(df)}")
        print(f"nb_groups \\in [{df.nb_groups.min()},{df.nb_groups.max()}]")
        print(f"seq_len \\in [{df.max_len_group.min()},{df.max_len_group.max()}]")
        print(f"total_len \\in [{df.total_len.min()},{df.total_len.max()}]")
        print()

    """
    Subset: train
    Dataset length: 205942
    nb_groups \'in [2,15]
    seq_len \'in [4,25]
    total_len \'in [12,227]

    Subset: val
    Dataset length: 39472
    nb_groups \'in [1,15]
    seq_len \'in [3,26]
    total_len \'in [5,227]

    Subset: test
    Dataset length: 78143
    nb_groups \\in [4,15]
    seq_len \\in [3,25]
    total_len \\in [16,213]
    """


if __name__ == '__main__':
    main()
