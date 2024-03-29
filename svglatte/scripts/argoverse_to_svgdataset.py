import argparse
import os
from concurrent import futures

import pandas as pd
import torch
from tqdm import tqdm

from deepsvg.dataset.preprocess import main as deepsvg_dataset_preprocess_main
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.geom import Bbox
from deepsvg.svglib.svg import SVG
from svglatte.dataset.argoverse_dataset import ArgoverseDataset, CMDS_CLASSES
from svglatte.utils.util import Object, ensure_dir


def svgtensor_data_to_svg_file(
        svgtensor_data,
        output_svg_path,
        meta_data,
        cache_format=ArgoverseDataset.SVGTENSOR_SEQUENCES_FORMAT,
        viewbox_size=255,
):
    filename = os.path.splitext(os.path.basename(output_svg_path))[0]
    if cache_format == ArgoverseDataset.GROUPED_SEQUENCES_FORMAT:
        seq = svgtensor_data
        cmds_grouped = torch.argmax(seq[..., :CMDS_CLASSES], dim=-1)
        args_grouped = seq[..., CMDS_CLASSES:]
        svgtensor = SVGTensor.from_cmd_args(cmds_grouped, args_grouped)
        svgtensor.unpad()  # removes eos (and padding, but the preprocessed sequence have no padding)
        svgtensor.drop_sos()
        svgtensor_data = svgtensor.data
    elif cache_format == ArgoverseDataset.SVGTENSOR_SEQUENCES_FORMAT:
        svgtensor_data = svgtensor_data
    else:
        raise RuntimeError(f"Unrecognized cache format: '{cache_format}'")

    svg = SVG.from_tensor(svgtensor_data, viewbox=Bbox(viewbox_size))
    svg.split_paths()
    svg.save_svg(output_svg_path)

    len_groups = [path_group.total_len() for path_group in svg.svg_path_groups]

    meta_data[filename] = {
        "id": filename,
        "total_len": sum(len_groups),
        "nb_groups": len(len_groups),
        "len_groups": len_groups,
        "max_len_group": max(len_groups)
    }


def argoverse_to_svg_dataset(caching_path_sequences, output_folder, max_workers):
    # Prepare arguments for DeepSVG's main
    args = Object()
    args.data_folder = os.path.join(output_folder, "svgs")
    args.data_meta_file = os.path.join(output_folder, "svg_meta_non_simplified.csv")
    args.output_folder = os.path.join(output_folder, "svgs_simplified")
    args.output_meta_file = os.path.join(output_folder, "svg_meta.csv")
    args.workers = max_workers
    ensure_dir(args.data_folder)
    ensure_dir(args.output_folder)

    # Load the svgtensors
    assert os.path.isfile(caching_path_sequences)
    svgtensor_data_list = torch.load(caching_path_sequences)

    # Create SVGs from svgtensors and save them to disk
    meta_data = {}
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(svgtensor_data_list)) as pbar:
            preprocess_requests = [
                executor.submit(svgtensor_data_to_svg_file, svgtensor_data, os.path.join(args.data_folder, f"{i}.svg"),
                                meta_data, )
                for i, svgtensor_data in enumerate(svgtensor_data_list)
            ]
            for _ in futures.as_completed(preprocess_requests):
                pbar.update(1)

    df = pd.DataFrame(meta_data.values())
    df.to_csv(args.data_meta_file, index=False)

    # Use DeepSVG's preprocessing functionality
    deepsvg_dataset_preprocess_main(args)


def main(args):
    """Convert Argoverse to a deepsvg.svg_dataset.SVGDataset so that it can be evaluated with DeepSVG."""
    argoverse_to_svg_dataset(
        caching_path_sequences=args.input_argoverse_subset_file,
        output_folder=args.output_deepsvg_format_subset_folder,
        max_workers=args.workers,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_argoverse_subset_file', type=str, default='data/argoverse/train.sequences.torchsave')
    parser.add_argument('--output_deepsvg_format_subset_folder', type=str, default='data/svgdataset/train')
    parser.add_argument('--workers', type=int, default=40)
    args = parser.parse_args()
    print(f"Args: {args}")

    # Create plots dir if it does not exist.
    ensure_dir(args.output_deepsvg_format_subset_folder)

    main(args)
