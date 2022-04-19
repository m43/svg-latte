import os

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from svglatte.dataset.deepsvg_dataset import DeepSVGDatasetNoCache

CMDS_CLASSES = 7
ARGS_DIM = 11
SEQ_FEATURE_DIM = CMDS_CLASSES + ARGS_DIM


class ArgoverseDataset(Dataset):
    def __init__(self, caching_path_prefix, rendered_images_width=64, rendered_images_height=64):
        self.caching_path_prefix = caching_path_prefix
        self.rendered_images_width = rendered_images_width
        self.rendered_images_height = rendered_images_height

        self.caching_path_sequences = f"{caching_path_prefix}.sequences.torchsave"
        self.caching_path_rendered_images = (
            f"{caching_path_prefix}.rendered_images.{rendered_images_width}x{rendered_images_height}.torchsave")
        self.caching_path_seq_mean = f"{caching_path_prefix}.seq_mean.torchsave"
        self.caching_path_seq_std = f"{caching_path_prefix}.seq_std.torchsave"

        assert os.path.isfile(self.caching_path_sequences)
        assert os.path.isfile(self.caching_path_rendered_images)
        assert os.path.isfile(self.caching_path_seq_mean)
        assert os.path.isfile(self.caching_path_seq_std)

        self.sequences = torch.load(self.caching_path_sequences)
        self.rendered_images = torch.load(self.caching_path_rendered_images)
        self.seq_mean = torch.load(self.caching_path_seq_mean)
        self.seq_std = torch.load(self.caching_path_seq_std)

        assert len(self.sequences) == len(self.rendered_images)

    def __getitem__(self, idx):
        length = torch.tensor(len(self.sequences[idx]))
        return self.sequences[idx], self.rendered_images[idx], length

    def __len__(self):
        return len(self.sequences)


class ArgoverseDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root,
            pad_val=-1,
            rendered_images_width=64,
            rendered_images_height=64,
            batch_size: int = 32,
            fast_run: bool = False,
    ):
        super(ArgoverseDataModule, self).__init__()
        self.batch_size = batch_size

        if fast_run:
            print("Fast run (train_ds = test_ds = val_ds)")
            self.val_ds = ArgoverseDataset(
                os.path.join(data_root, f"val"),
                rendered_images_width,
                rendered_images_height
            )
            self.train_ds = self.test_ds = self.val_ds
        else:
            self.train_ds = ArgoverseDataset(
                os.path.join(data_root, f"train"),
                rendered_images_width,
                rendered_images_height
            )
            self.val_ds = ArgoverseDataset(
                os.path.join(data_root, f"val"),
                rendered_images_width,
                rendered_images_height
            )
            self.test_ds = ArgoverseDataset(
                os.path.join(data_root, f"test"),
                rendered_images_width,
                rendered_images_height
            )

        def collate_fn(batch):
            return DeepSVGDatasetNoCache.pad_collate_fn(batch, pad_val)

        self.collate_fn = collate_fn

        self.train_mean = self.train_ds.seq_mean
        self.train_mean[:CMDS_CLASSES] = 0.
        self.train_std = self.train_ds.seq_std
        self.train_std[:CMDS_CLASSES] = 1.

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=0)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=0)
