import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svgtensor_dataset import SVGTensorDataset

CMDS_CLASSES = 7
ARGS_DIM = 11
SEQ_FEATURE_DIM = CMDS_CLASSES + ARGS_DIM


class DeepSVGDataset(Dataset):
    def __init__(
            self,
            svgtensor_dataset: SVGTensorDataset,
            data_root=None,
            cache_to_disk=True,
            clean_disk_cache=False,
            #             num_processes=2,
    ):
        self.svgtensor_dataset = svgtensor_dataset

    #         self.images_cached_path = os.path.join(data_root, f"images.torchsave") if cache_to_disk else None
    #         if cache_to_disk and not clean_disk_cache and os.path.isfile(self.images_cached_path):
    #             self.sequences, self.rendered_images = torch.load(self.images_cached_path)
    #             return

    #         with Pool(num_processes) as p:
    #             length = math.ceil(len(self.svgtensor_dataset) / num_processes)
    #             svgtensor_list = [(self.svgtensor_dataset, i, length) for i in range(num_processes)]
    #             self.rendered_images = torch.cat(p.map(DeepSVGDataset._pool_f, svgtensor_list))

    #         self.rendered_images = torch.zeros((len(self.svgtensor_dataset), 1, 64, 64), dtype=torch.float32)
    #         for i, batch in enumerate(tqdm.tqdm(self.svgtensor_dataset)):
    #             self.rendered_images[i, 0, :, :] = DeepSVGDataset.cmd_arg_to_img(batch["commands_grouped"][0], batch["args_grouped"][0])

    #         if cache_to_disk:
    #             torch.save(self.rendered_images, self.images_cached_path)

    #     @staticmethod
    #     def _pool_f(x):
    #         svgtensor_dataset, idx, length = x
    #         start_idx = idx*length
    #         end_idx = min(len(svgtensor_dataset), (idx+1)*length)
    #         assert end_idx > start_idx

    #         rendered_images = torch.zeros((end_idx-start_idx, 1, 64, 64), dtype=torch.float32)
    #         for i in tqdm.tqdm(range(start_idx, end_idx)):
    #             batch = svgtensor_dataset[i]
    #             rendered_images[i-start_idx, 0, :, :] = DeepSVGDataset.cmd_arg_to_img(batch["commands_grouped"][0], batch["args_grouped"][0])
    #         return rendered_images

    #         rendered_images = torch.zeros((len(svgtensor), 1, 64, 64), dtype=torch.float32)
    #         for i, batch in enumerate(tqdm.tqdm(svgtensor)):
    #             rendered_images[i, 0, :, :] = DeepSVGDataset.cmd_arg_to_img(batch["commands_grouped"][0], batch["args_grouped"][0])
    #         return rendered_images

    @staticmethod
    def cmd_arg_to_img(cmds, args):
        svgtensor = SVGTensor.from_cmd_args(cmds, args).unpad().drop_sos()
        pil_image = svgtensor.draw(do_display=False, return_png=True)
        pil_image = ImageOps.grayscale(pil_image)
        return torch.tensor(np.array(pil_image)).unsqueeze(0) / 255.

    @staticmethod
    def cmds_args_to_seq(cmds, args):
        cmds_onehot = torch.nn.functional.one_hot(cmds.to(torch.int64), num_classes=CMDS_CLASSES)
        return torch.cat((cmds_onehot, args), dim=1)

    def __getitem__(self, idx):
        item = self.svgtensor_dataset[idx]
        cmds, args = item["commands_grouped"][0], item["args_grouped"][0]
        seq = DeepSVGDataset.cmds_args_to_seq(cmds, args)
        img = DeepSVGDataset.cmd_arg_to_img(cmds, args)
        # img = self.rendered_images[idx]
        return seq, img

    def __len__(self):
        return len(self.svgtensor_dataset)


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

        train_val_df, test_df = train_test_split(df, test_size=test_length, random_state=seed)
        train_df, val_df = train_test_split(train_val_df, test_size=val_length, random_state=seed)

        train_df.to_csv(train_df_cache_path)
        val_df.to_csv(val_df_cache_path)
        test_df.to_csv(test_df_cache_path)
    else:
        train_df = pd.load_csv(train_df_cache_path)
        val_df = pd.load_csv(val_df_cache_path)
        test_df = pd.load_csv(test_df_cache_path)

    model_args = ["commands_grouped", "args_grouped"]
    train_svgtensor_ds = SVGTensorDataset(train_df, data_root, model_args, max_num_groups, max_seq_len, max_total_len)
    val_svgtensor_ds = SVGTensorDataset(val_df, data_root, model_args, max_num_groups, max_seq_len, max_total_len)
    test_svgtensor_ds = SVGTensorDataset(test_df, data_root, model_args, max_num_groups, max_seq_len, max_total_len)

    train_ds = DeepSVGDataset(train_svgtensor_ds, data_root, cache_to_disk, clean_disk_cache)
    val_ds = DeepSVGDataset(val_svgtensor_ds, data_root, cache_to_disk, clean_disk_cache)
    test_ds = DeepSVGDataset(test_svgtensor_ds, data_root, cache_to_disk, clean_disk_cache)

    return train_ds, val_ds, test_ds


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

        self.train_ds, self.val_ds, self.test_ds = load_dataset_splits(
            data_root=data_root,
            meta_filepath=meta_filepath,
            max_num_groups=max_num_groups,
            max_seq_len=max_seq_len,
            max_total_len=max_total_len,
            cache_to_disk=cache_to_disk,
            clean_disk_cache=clean_disk_cache
        )

        self.train_mean_and_std_cached_path = f"{data_root}/train_mean_and_std.torchsave" if cache_to_disk else None
        if cache_to_disk and not clean_disk_cache and os.path.isfile(self.train_mean_and_std_cached_path):
            self.train_mean, self.train_std = torch.load(self.train_mean_and_std_cached_path)
            return

        # TODO
        #         train_sequences = self.train_ds[:]
        #         train_sequences = train_sequences.reshape(-1, train_sequences.shape[-1])
        #         not_eos_or_padding_indices = (train_sequences[:, 0] == 0.) * (train_sequences[:, :CMDS_CLASSES].sum(dim=1) != 0.)
        #         train_sequences = train_sequences[not_eos_or_padding_indices]
        self.train_mean = torch.zeros(SEQ_FEATURE_DIM)  # train_sequences.mean(dim=0)
        self.train_mean[:CMDS_CLASSES] = 0.
        self.train_std = torch.ones(SEQ_FEATURE_DIM)  # train_sequences.std(dim=0)
        self.train_std[:CMDS_CLASSES] = 1.

    #         if cache_to_disk:
    #             torch.save((self.train_mean, self.train_std), self.train_mean_and_std_cached_path)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=20)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=20)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4)
