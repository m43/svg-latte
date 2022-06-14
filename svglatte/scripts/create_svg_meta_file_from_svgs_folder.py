import glob
import logging
import os
from argparse import ArgumentParser
from concurrent import futures

import pandas as pd
from tqdm import tqdm

from deepsvg.svglib.svg import SVG


def preprocess_svg(svg_file, meta_data):
    filename = os.path.splitext(os.path.basename(svg_file))[0]

    svg = SVG.load_svg(svg_file)
    len_groups = [path_group.total_len() for path_group in svg.svg_path_groups]
    meta_data[filename] = {
        "id": filename,
        "total_len": sum(len_groups),
        "nb_groups": len(len_groups),
        "len_groups": len_groups,
        "max_len_group": max(len_groups)
    }


def main(args):
    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        svg_files = glob.glob(os.path.join(args.data_folder, "*.svg"))
        meta_data = {}

        with tqdm(total=len(svg_files)) as pbar:
            preprocess_requests = [executor.submit(preprocess_svg, svg_file, meta_data)
                                   for svg_file in svg_files]

            for _ in futures.as_completed(preprocess_requests):
                pbar.update(1)

    df = pd.DataFrame(meta_data.values())
    df.to_csv(args.output_meta_file, index=False)

    logging.info("SVG Preprocessing complete.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--data_folder", default=os.path.join("dataset", "svgs"))
    parser.add_argument("--output_meta_file", default=os.path.join("dataset", "svg_meta.csv"))
    parser.add_argument("--workers", default=4, type=int)

    args = parser.parse_args()

    main(args)
