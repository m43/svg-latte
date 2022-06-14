import argparse
import os
import pandas as pd


def main(args):
    """
    Count the real maximum values for max_num_groups, max_seq_len, and max_total_len.
    They might be lower than the numbers used when creating Argoverse in SVG-Net, i.e.
        self.max_num_groups = 120
        self.max_seq_len = 200
        self.max_total_len = 2000
    If the numbers are really lower, then it will be easier to evaluate them with DeepSVG.
    """
    for subset in ["train", "val", "test"]:
        csv_path = os.path.join(args.svgdataset_path, f"{subset}/svg_meta.csv")

        print(f"Subset: {subset}")
        df = pd.read_csv(csv_path)
        print(f"Dataset length: {len(df)}")
        print(f"nb_groups \\in [{df.nb_groups.min()},{df.nb_groups.max()}]")
        print(f"seq_len \\in [{df.max_len_group.min()},{df.max_len_group.max()}]")
        print(f"total_len \\in [{df.total_len.min()},{df.total_len.max()}]")
        print()

    """
    Argo-v4 (different preprocessing, SVGs are more complex)

    Subset: train
    Dataset length: 205942
    nb_groups \in [1,99]
    seq_len \in [6,35]
    total_len \in [17,672]

    Subset: val
    Dataset length: 39472
    nb_groups \in [4,99]
    seq_len \in [5,33]
    total_len \in [31,625]

    Subset: test
    Dataset length: 78138
    nb_groups \in [1,99]
    seq_len \in [2,31]
    total_len \in [2,631]

    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--svgdataset_path', type=str, default="data/argoverse/svgdataset/")

    args = parser.parse_args()
    print(f"Args: {args}")

    main(args)
