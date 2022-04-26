import os

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from svglatte.dataset.deepsvg_dataset import DeepSVGDatasetNoCache

CMDS_CLASSES = 7
ARGS_DIM = 11
SEQ_FEATURE_DIM = CMDS_CLASSES + ARGS_DIM


class ArgoverseDataset(Dataset):
    def __init__(
            self,
            caching_path_prefix,
            rendered_images_width=64,
            rendered_images_height=64,
            render_on_the_fly=False,
    ):
        self.caching_path_prefix = caching_path_prefix
        self.rendered_images_width = rendered_images_width
        self.rendered_images_height = rendered_images_height
        self.render_on_the_fly = render_on_the_fly

        self.caching_path_sequences = f"{caching_path_prefix}.sequences.torchsave"
        self.caching_path_seq_mean = f"{caching_path_prefix}.seq_mean.torchsave"
        self.caching_path_seq_std = f"{caching_path_prefix}.seq_std.torchsave"
        if not render_on_the_fly:
            self.caching_path_rendered_images = (
                f"{caching_path_prefix}.rendered_images.{rendered_images_width}x{rendered_images_height}.torchsave")

        assert os.path.isfile(self.caching_path_sequences)
        assert os.path.isfile(self.caching_path_seq_mean)
        assert os.path.isfile(self.caching_path_seq_std)
        if not render_on_the_fly:
            assert os.path.isfile(self.caching_path_rendered_images)

        self._sequences = torch.load(self.caching_path_sequences)
        self._seq_mean = torch.load(self.caching_path_seq_mean)
        self._seq_std = torch.load(self.caching_path_seq_std)
        if not render_on_the_fly:
            self._rendered_images = torch.load(self.caching_path_rendered_images)
            assert len(self._sequences) == len(self._rendered_images)


    @staticmethod
    def draw_svgtensor(
            svgtensor,
            output_width,
            output_height,
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
        pil_image = ArgoverseDataset.draw_svgtensor(svgtensor, output_width, output_height)
        return torch.tensor(np.array(pil_image).transpose(2, 0, 1)[-1:]) / 255.

    def __getitem__(self, idx):
        seq = self._sequences[idx]
        length = torch.tensor(len(seq))
        if self.render_on_the_fly:
            rendered_image = self._render_on_the_fly(idx, seq)
        else:
            rendered_image = self._rendered_images[idx]
        return seq, rendered_image, length

    def __len__(self):
        return len(self._sequences)

    def get_number_of_sequence_dimensions(self):
        return len(self.SEQUENCE_FEATURE_DIMENSTIONS)

    def _render_on_the_fly(self, idx, seq):
        # if self.cache_render_on_the_fly and self.rendered_images[idx] is not None:
        #     return self.rendered_images[idx]

        cmds = torch.argmax(seq[..., :CMDS_CLASSES], dim=-1)
        args = seq[..., CMDS_CLASSES:]
        rendered_image = ArgoverseDataset.cmd_arg_to_img(
            cmds, args,
            self.rendered_images_width,
            self.rendered_images_height
        )

        # if self.cache_render_on_the_fly:
        #     self.rendered_images[idx] = rendered_image

        return rendered_image



class ArgoverseDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root,
            pad_val=-1,
            rendered_images_width=64,
            rendered_images_height=64,
            batch_size: int = 32,
            render_on_the_fly: bool = False,
            fast_run: bool = False,
    ):
        super(ArgoverseDataModule, self).__init__()
        self.batch_size = batch_size

        if fast_run:
            print("Fast run (train_ds = test_ds = val_ds)")
            self.val_ds = ArgoverseDataset(
                os.path.join(data_root, f"val"),
                rendered_images_width,
                rendered_images_height,
                render_on_the_fly,
            )
            self.train_ds = self.test_ds = self.val_ds
        else:
            self.train_ds = ArgoverseDataset(
                os.path.join(data_root, f"train"),
                rendered_images_width,
                rendered_images_height,
                render_on_the_fly,
            )
            self.val_ds = ArgoverseDataset(
                os.path.join(data_root, f"val"),
                rendered_images_width,
                rendered_images_height,
                render_on_the_fly,
            )
            self.test_ds = ArgoverseDataset(
                os.path.join(data_root, f"test"),
                rendered_images_width,
                rendered_images_height,
                render_on_the_fly,
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
