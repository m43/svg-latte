import os
import pickle

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms as T

CMDS_CLASSES = 4
ARGS_DIM = 6
SEQ_FEATURE_DIM = CMDS_CLASSES + ARGS_DIM


class DeepVecFontDataset(Dataset):
    def __init__(
            self,
            pkl_path,
            max_seq_len,
            seq_feature_dim,
            glyphs_in_alphabet,
            rendered_transform=T.Compose([T.Lambda(lambda x: 1. - x)]),
            cache_to_disk=True,
            clean_disk_cache=False,
    ):
        self.glyphs_in_alphabet = glyphs_in_alphabet
        self.max_seq_len = max_seq_len
        self.seq_feature_dim = seq_feature_dim
        self.cached_path = f"{pkl_path}.torchsave" if cache_to_disk else None

        if cache_to_disk and not clean_disk_cache and os.path.isfile(self.cached_path):
            self.sequences, self.rendered_images = torch.load(self.cached_path)
            return

        with open(pkl_path, "rb") as pkl_f:
            all_glyphs = pickle.load(pkl_f)

        # TODO perhaps the shapes should be different for efficiency during forward pass
        self.sequences = torch.tensor([
            alphabet["sequence"][i]
            for alphabet in all_glyphs
            for i in range(self.glyphs_in_alphabet)
        ]).reshape(-1, self.max_seq_len, self.seq_feature_dim)
        self.rendered_images = torch.tensor([
            alphabet["rendered"][i]
            for alphabet in all_glyphs
            for i in range(self.glyphs_in_alphabet)
        ]).reshape(-1, 1, 64, 64) / 255.
        self.rendered_images = rendered_transform(self.rendered_images)

        assert len(self.sequences) == len(self.rendered_images)

        if cache_to_disk:
            torch.save((self.sequences, self.rendered_images), self.cached_path)

    def __getitem__(self, idx):
        return self.sequences[idx], self.rendered_images[idx]

    def __len__(self):
        return len(self.sequences)


class DeepVecFontDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root,
            max_seq_len,
            seq_feature_dim,
            glyphs_in_alphabet=52,
            valid_ratio=0.2,
            batch_size: int = 32,
            cache_to_disk=True,
            clean_disk_cache=False,
    ):
        super(DeepVecFontDataModule, self).__init__()
        self.batch_size = batch_size

        self.fonts_train = DeepVecFontDataset(
            f"{data_root}/train_all.pkl",
            max_seq_len=max_seq_len,
            seq_feature_dim=seq_feature_dim,
            glyphs_in_alphabet=glyphs_in_alphabet,
            cache_to_disk=cache_to_disk,
            clean_disk_cache=clean_disk_cache,
        )
        n = len(self.fonts_train)
        n_val = int(n * valid_ratio)
        self.fonts_train, self.fonts_val = random_split(self.fonts_train, lengths=[n - n_val, n_val])
        print(f"First idx in train: {self.fonts_train.indices[0]}\nFirst idx in val: {self.fonts_val.indices[0]}")
        self.fonts_test = DeepVecFontDataset(
            f"{data_root}/test_all.pkl",
            max_seq_len=max_seq_len,
            seq_feature_dim=seq_feature_dim,
            glyphs_in_alphabet=glyphs_in_alphabet,
            cache_to_disk=cache_to_disk,
            clean_disk_cache=clean_disk_cache,
        )

        self.train_mean_and_std_cached_path = f"{data_root}/train_mean_and_std.torchsave" if cache_to_disk else None
        if cache_to_disk and not clean_disk_cache and os.path.isfile(self.train_mean_and_std_cached_path):
            self.train_mean, self.train_std = torch.load(self.train_mean_and_std_cached_path)
            return

        train_sequences = self.fonts_train.dataset[self.fonts_train.indices][0]
        train_sequences = train_sequences.reshape(-1, train_sequences.shape[-1])
        not_eos_or_padding_indices = (train_sequences[:, 0] == 0.) * (train_sequences[:, :4].sum(dim=1) != 0.)
        train_sequences = train_sequences[not_eos_or_padding_indices]
        self.train_mean = train_sequences.mean(dim=0)
        self.train_mean[:4] = 0.
        self.train_std = train_sequences.std(dim=0)
        self.train_std[:4] = 1.
        # self.train_mean = np.load(f'{data_root}/mean.npz')
        # self.train_std = np.load(f'{data_root}/stdev.npz')

        if cache_to_disk:
            torch.save((self.train_mean, self.train_std), self.train_mean_and_std_cached_path)

    def train_dataloader(self):
        return DataLoader(self.fonts_train, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.fonts_val, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.fonts_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.fonts_test, batch_size=self.batch_size)
