import io
import os

import cairosvg
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.geom import Bbox
from deepsvg.svglib.svg import SVGPath, SVG
from deepsvg.svgtensor_dataset import SVGTensorDataset
from svglatte.utils.util import pad_collate_fn

CMDS_CLASSES = 7
ARGS_DIM = 11
SEQ_FEATURE_DIM = CMDS_CLASSES + ARGS_DIM

x = False


class DeepSVGDatasetNoCache(Dataset):
    def __init__(self, svgtensor_dataset: SVGTensorDataset):
        self.svgtensor_dataset = svgtensor_dataset
        self.seq_mean = torch.zeros(SEQ_FEATURE_DIM, dtype=torch.float32)
        self.seq_std = torch.ones(SEQ_FEATURE_DIM, dtype=torch.float32)

    @staticmethod
    def draw_svgtensor(
            svgtensor,
            output_width=64,
            output_height=64,
            fill=False,
            with_points=False,
            with_handles=False,
            with_bboxes=False,
            with_markers=False,
            color_firstlast=False,
            with_moves=True,
    ):
        svgpath = SVGPath.from_tensor(svgtensor.data)
        svg = SVG([svgpath], viewbox=Bbox(24))

        svg_str = svg.to_str(fill=fill, with_points=with_points, with_handles=with_handles, with_bboxes=with_bboxes,
                             with_markers=with_markers, color_firstlast=color_firstlast, with_moves=with_moves)

        img_data = cairosvg.svg2png(
            bytestring=svg_str,
            invert_images=True,
            output_width=output_width,
            output_height=output_height,
        )
        return Image.open(io.BytesIO(img_data))

    @staticmethod
    def cmd_arg_to_img(cmds, args, output_width=64, output_height=64):
        svgtensor = SVGTensor.from_cmd_args(cmds, args).unpad().drop_sos()
        pil_image = DeepSVGDatasetNoCache.draw_svgtensor(svgtensor, output_width, output_height)
        global x
        if x == False:
            x = True
            print(pil_image)
            print(torch.tensor(np.array(pil_image).transpose(2, 0, 1)[0]))
            print(torch.tensor(np.array(pil_image).transpose(2, 0, 1)[1]))
            print(torch.tensor(np.array(pil_image).transpose(2, 0, 1)[2]))
            print(torch.tensor(np.array(pil_image).transpose(2, 0, 1)[3]))
            print(torch.tensor(np.array(pil_image).transpose(2, 0, 1)[-1:]).shape)
        return torch.tensor(np.array(pil_image).transpose(2, 0, 1)[-1:]) / 255.

    @staticmethod
    def cmds_args_to_seq(cmds, args):
        cmds_onehot = torch.nn.functional.one_hot(cmds.to(torch.int64), num_classes=CMDS_CLASSES)
        return torch.cat((cmds_onehot, args), dim=1)

    def __getitem__(self, idx):
        item = self.svgtensor_dataset[idx]
        cmds, args, seq_len = item["commands_grouped"][0], item["args_grouped"][0], item["total_len"]
        cmds, args = cmds[:seq_len + 2], args[:seq_len + 2]  # +2 in order to include SOS and EOS
        seq = DeepSVGDatasetNoCache.cmds_args_to_seq(cmds, args)
        img = DeepSVGDatasetNoCache.cmd_arg_to_img(cmds, args)
        return seq, img, seq_len + 2

    def __len__(self):
        return len(self.svgtensor_dataset)


class DeepSVGDataset(Dataset):
    def __init__(
            self,
            svgtensor_dataset: SVGTensorDataset,
            cache_to_disk=True,
            clean_disk_cache=False,
            caching_path_prefix=None,
            caching_batch_size=512,
            caching_num_workers=40,
    ):

        self.caching_path_sequences = f"{caching_path_prefix}.sequences.torchsave" if cache_to_disk else None
        self.caching_path_rendered_images = f"{caching_path_prefix}.rendered_images.torchsave" if cache_to_disk else None
        self.caching_path_seq_mean = f"{caching_path_prefix}.seq_mean.torchsave" if cache_to_disk else None
        self.caching_path_seq_std = f"{caching_path_prefix}.seq_std.torchsave" if cache_to_disk else None
        cache_exists = (
                os.path.isfile(self.caching_path_sequences)
                and os.path.isfile(self.caching_path_rendered_images)
                and os.path.isfile(self.caching_path_seq_mean)
                and os.path.isfile(self.caching_path_seq_std)
        )

        if cache_to_disk and not clean_disk_cache and cache_exists:
            self.sequences = torch.load(self.caching_path_sequences)
            self.rendered_images = torch.load(self.caching_path_rendered_images)
            self.seq_mean = torch.load(self.caching_path_seq_mean)
            self.seq_std = torch.load(self.caching_path_seq_std)
            return

        print(f"CACHING LOG: CACHING OF DATASET STARTED, caching_path_prefix={caching_path_prefix}")
        dataset = DeepSVGDatasetNoCache(svgtensor_dataset)
        print(f"CACHING LOG: len(dataset)={len(dataset)}")

        # Infer the shape
        print("CACHING LOG: Inferring shape")
        seq, img = dataset[0]
        sequences_shape = (len(dataset),) + seq.shape
        rendered_images_shape = (len(dataset),) + img.shape

        # Big tensors and dataset iterator
        # Hope we survive the next few lines :-) Ok we survived on a 700GB RAM node and with some hacking (hack:
        # caching one dataset after the other because os.fork would fork the previous gigantic dataset if it had
        # already been loaded to memory)
        print("CACHING LOG:"
              " Creating the dataloader and iterator (iterator forces os.fork() before big tensors are created)")
        dataloader = DataLoader(dataset, batch_size=caching_batch_size, num_workers=caching_num_workers)
        dataloader_iter = iter(dataloader)

        print("CACHING LOG: Creating the big tensors")
        self.sequences = seq.new_ones(sequences_shape)
        self.rendered_images = img.new_ones(rendered_images_shape)

        print(f"CACHING LOG: Tensor shapes and dtypes:"
              f"\n  sequences.shape={self.sequences.shape}"
              f"\n  sequences.dtype={self.sequences.dtype}"
              f"\n  rendered_images.shape={self.rendered_images.shape}"
              f"\n  rendered_images.dtype={self.rendered_images.dtype}")
        # torch.save(self.sequences, self.caching_path_sequences)

        # Fill the big tensors
        print("CACHING LOG: Filling the big tensors by iterating the dataloader")
        for batch_idx, (seq, img) in enumerate(tqdm.tqdm(dataloader_iter)):
            start_pos = batch_idx * caching_batch_size
            end_pos = batch_idx * caching_batch_size + len(seq)
            self.sequences[start_pos:end_pos] = seq
            self.rendered_images[start_pos:end_pos] = img

        # Compute mean and std of sequences
        print("CACHING LOG: Computing sequence mean and std")
        sequences = self.sequences.view(-1, self.sequences.shape[-1])
        #                         0    1    2    3     4      5     6
        # COMMANDS_SIMPLIFIED = ["m", "l", "c", "a", "EOS", "SOS", "z"]
        not_eos_or_sos_or_pad_indices = sequences[:, :4].sum(dim=1) != 0.
        sequences = sequences[not_eos_or_sos_or_pad_indices]
        self.seq_mean = sequences.mean(dim=0)
        self.seq_mean[:CMDS_CLASSES] = 0.
        self.seq_std = sequences.std(dim=0)
        self.seq_std[:CMDS_CLASSES] = 1.

        # Saving to disk
        print("CACHING LOG: Saving to disk")
        if cache_to_disk:
            torch.save(self.sequences, self.caching_path_sequences)
            torch.save(self.rendered_images, self.caching_path_rendered_images)
            torch.save(self.seq_mean, self.caching_path_seq_mean)
            torch.save(self.seq_std, self.caching_path_seq_std)
        print("CACHING LOG: Caching done")

    def __getitem__(self, idx):
        return self.sequences[idx], self.rendered_images[idx]

    def __len__(self):
        return len(self.sequences)


def load_dataset_splits(
        data_root,
        meta_filepath,
        max_num_groups,
        max_seq_len,
        max_total_len=None,
        val_ratio=0.15,
        test_ratio=0.15,
        cache_to_disk=True,
        clean_disk_cache=False,
        seed=72
):
    train_df_cache_path = os.path.join(data_root, f"train_df.csv") if cache_to_disk else None
    val_df_cache_path = os.path.join(data_root, f"val_df.csv") if cache_to_disk else None
    test_df_cache_path = os.path.join(data_root, f"test_df.csv") if cache_to_disk else None
    cached_exists = (
            cache_to_disk
            and os.path.isfile(train_df_cache_path)
            and os.path.isfile(val_df_cache_path)
            and os.path.isfile(test_df_cache_path)
    )

    if cache_to_disk or clean_disk_cache or not cached_exists:
        df = pd.read_csv(meta_filepath)

        df = df[df.nb_groups <= max_num_groups]
        df = df[df.max_len_group <= max_seq_len]
        if max_total_len is not None:
            df = df[df.total_len <= max_total_len]

        assert val_ratio + test_ratio < 1.0
        val_length = int(len(df) * val_ratio)
        test_length = int(len(df) * test_ratio)

        # TODO
        raise Exception("loading train_test_split from sklearn uses a lot of memory which"
                        " could be a problem for torch dataloader workers")
        from sklearn.model_selection import train_test_split
        train_val_df, test_df = train_test_split(df, test_size=test_length, random_state=seed)
        train_df, val_df = train_test_split(train_val_df, test_size=val_length, random_state=seed)

        train_df.to_csv(train_df_cache_path)
        val_df.to_csv(val_df_cache_path)
        test_df.to_csv(test_df_cache_path)
    else:
        train_df = pd.load_csv(train_df_cache_path)
        val_df = pd.load_csv(val_df_cache_path)
        test_df = pd.load_csv(test_df_cache_path)

    model_args = ["total_len", "commands_grouped", "args_grouped"]
    train_svgtensor_ds = SVGTensorDataset(train_df, data_root, model_args, max_num_groups, max_seq_len, max_total_len)
    val_svgtensor_ds = SVGTensorDataset(val_df, data_root, model_args, max_num_groups, max_seq_len, max_total_len)
    test_svgtensor_ds = SVGTensorDataset(test_df, data_root, model_args, max_num_groups, max_seq_len, max_total_len)

    ## Without caching
    # train_ds = DeepSVGDataset(train_svgtensor_ds, cache_to_disk, clean_disk_cache,
    #                           os.path.join(data_root, f"train"))
    # val_ds = DeepSVGDataset(val_svgtensor_ds, cache_to_disk, clean_disk_cache,
    #                         os.path.join(data_root, f"val"))
    # test_ds = DeepSVGDataset(test_svgtensor_ds, cache_to_disk, clean_disk_cache,
    #                          os.path.join(data_root, f"test"))
    # collate_fn = torch.utils.data.default_collate

    ## With caching
    train_ds = DeepSVGDatasetNoCache(train_svgtensor_ds)
    val_ds = DeepSVGDatasetNoCache(val_svgtensor_ds)
    test_ds = DeepSVGDatasetNoCache(test_svgtensor_ds)

    def collate_fn(batch):
        return pad_collate_fn(batch, train_svgtensor_ds.PAD_VAL)

    return train_ds, val_ds, test_ds, collate_fn


class DeepSVGDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root,
            meta_filepath,
            max_num_groups,
            max_seq_len,
            max_total_len,
            batch_size: int = 32,
            cache_to_disk=True,
            clean_disk_cache=False,
    ):
        super(DeepSVGDataModule, self).__init__()
        self.batch_size = batch_size

        self.train_ds, self.val_ds, self.test_ds, self.collate_fn = load_dataset_splits(
            data_root=data_root,
            meta_filepath=meta_filepath,
            max_num_groups=max_num_groups,
            max_seq_len=max_seq_len,
            max_total_len=max_total_len,
            cache_to_disk=cache_to_disk,
            clean_disk_cache=clean_disk_cache
        )

        self.train_mean = self.train_ds.seq_mean
        self.train_std = self.train_ds.seq_std

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=40)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=0)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=0)
