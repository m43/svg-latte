import io
import os
import random
from concurrent import futures
from datetime import datetime

import cairosvg
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from deepsvg.dataset.preprocess import main
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.geom import Bbox, Angle, Point
from deepsvg.svglib.svg import SVG
from svglatte.utils.util import pad_collate_fn, Object, ensure_dir

CMDS_CLASSES = 7
ARGS_DIM = 13  # 11 + 2


# SEQ_FEATURE_DIM = CMDS_CLASSES + ARGS_DIM

class ArgoverseDataset(Dataset):
    def __init__(
            self,
            caching_path_prefix,
            rendered_images_width=64,
            rendered_images_height=64,
            render_on_the_fly=True,
            remove_redundant_features=True,
            numericalize=False,
            zoom_preprocess_factor=1.0,  # zoom factor independent of augmentations
            augment=True,
            augment_shear_degrees=20,
            augment_rotate_degrees=180,
            augment_scale_min=0.6,
            augment_scale_max=1.1,
            augment_translate=5.4,
    ):
        self.render_on_the_fly = render_on_the_fly
        self.rendered_images_width = rendered_images_width
        self.rendered_images_height = rendered_images_height

        self.zoom_preprocess_factor = zoom_preprocess_factor
        self.remove_redundant_features = remove_redundant_features
        self.numericalize = numericalize

        self.augment = augment
        self.augment_rotate_degrees = augment_rotate_degrees
        self.augment_translate = augment_translate
        self.augment_scale_min = augment_scale_min
        self.augment_scale_max = augment_scale_max
        self.augment_shear_degrees = augment_shear_degrees

        self.caching_path_prefix = caching_path_prefix
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

        if self.remove_redundant_features:
            # Remove redundant features:
            # 1. DeepSVG has unused features in its standard format,
            # 2. Argoverse has only lines
            # Relevant fetaures:
            # - 0 is move
            # - 1 is line
            # - 4 is eos
            # - 5 is sos
            # - 12 is the start point x coordinate of the line
            # - 14 is the start point y coordinate of the line
            # - 16+2=18 is the end point x coordinate of the line
            # - 17+2=19 is the end point y coordinate of the line
            self.SEQUENCE_FEATURE_DIMENSTIONS = [0, 1, 4, 5, 12, 13, 16 + 2, 17 + 2]
            # Sanity check:
            # torch.cat([s.sum(0)[None, :] for s in self.sequences]).sum(0)
            # tensor([512979., 2584927., 0., 0., 39472., 39472.,
            #         0., -78944., -78944., -78944., -78944., -78944., START_POS_X, START_POS_Y,
            #         -3176850., -3176850., -3176850., -3176850., 28979378., 37385944.])
        else:
            self.SEQUENCE_FEATURE_DIMENSTIONS = range(CMDS_CLASSES + ARGS_DIM)

    @staticmethod
    def draw_svgtensor(
            svg,
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
    def svgtensor_to_img(svg, output_width=64, output_height=64):
        pil_image = ArgoverseDataset.draw_svgtensor(svg, output_width, output_height)
        return torch.tensor(np.array(pil_image).transpose(2, 0, 1)[-1:]) / 255.

    def __getitem__(self, idx):
        seq = self._sequences[idx]  # N x 7+11
        debug_length = torch.tensor(len(seq))

        # seq (pytorch tensor) to svgtensor (instance of SVGTensor)
        cmds = torch.argmax(seq[..., :CMDS_CLASSES], dim=-1)
        args = seq[..., CMDS_CLASSES:]
        svgtensor = SVGTensor.from_cmd_args(cmds, args)
        svgtensor.unpad()  # removes eos (and padding, but the preprocessed sequence have no padding)
        svgtensor.drop_sos()

        # svgtensor to svg (because svg has the easily accessible augmentation functionality (zoom, translate, ...)
        svg = SVG.from_tensor(svgtensor.data, viewbox=Bbox(24))

        # working in the svg domain (allows for simpler preprocessing and rendering compared to seq or svgtensor)
        svg = self._preprocess(svg)
        if self.render_on_the_fly:
            rendered_image = self._render_on_the_fly(idx, svg)
        else:
            rendered_image = self._rendered_images[idx]

        # svg back to svgtensor
        tensor_data = svg.to_tensor(concat_groups=True, PAD_VAL=-1)
        svgtensor = SVGTensor.from_data(tensor_data)

        # svgtensor back to seq
        svgtensor.add_eos()
        svgtensor.add_sos()
        cmds = svgtensor.cmds()
        cmds_onehot = torch.nn.functional.one_hot(cmds.to(torch.int64), num_classes=CMDS_CLASSES)
        args = svgtensor.args(with_start_pos=True)
        seq = torch.cat((cmds_onehot, args), dim=1)  # N x 7+13!
        length = torch.tensor(len(seq))
        assert length == debug_length
        assert CMDS_CLASSES + ARGS_DIM == len(seq[0])

        seq = seq[:, self.SEQUENCE_FEATURE_DIMENSTIONS]
        return seq, rendered_image, length

    def __len__(self):
        return len(self._sequences)

    def get_number_of_sequence_dimensions(self):
        return len(self.SEQUENCE_FEATURE_DIMENSTIONS)

    def _render_on_the_fly(self, idx, svg):
        # if self.cache_render_on_the_fly and self.rendered_images[idx] is not None:
        #     return self.rendered_images[idx]

        rendered_image = ArgoverseDataset.svgtensor_to_img(
            svg,
            self.rendered_images_width,
            self.rendered_images_height
        )

        # if self.cache_render_on_the_fly:
        #     self.rendered_images[idx] = rendered_image

        return rendered_image

    def get_sequences_mean(self):
        return torch.tensor([0])
        return self._seq_mean[self.SEQUENCE_FEATURE_DIMENSTIONS]

    def get_sequences_std(self):
        return torch.tensor([1])
        return self._seq_std[self.SEQUENCE_FEATURE_DIMENSTIONS]

    def _augment(self, svg):
        if self.augment_shear_degrees != 0.0:
            shear_angle = self.augment_shear_degrees * random.random()
            svg.shear(Angle(shear_angle))

        if self.augment_rotate_degrees != 0.0:
            rotation_angle = self.augment_rotate_degrees * (random.random() - 0.5) * 2
            svg.rotate(Angle(rotation_angle))

        zoom_factor = (self.augment_scale_max - self.augment_scale_min) * random.random() + self.augment_scale_min
        svg.zoom(zoom_factor)

        if self.augment_translate != 0.0:
            dx = self.augment_translate * (random.random() - 0.5) * 2
            dy = self.augment_translate * (random.random() - 0.5) * 2
            svg.translate(Point(dx, dy))

        return svg

    def _preprocess(self, svg):
        if self.zoom_preprocess_factor != 1.0:
            svg.zoom(self.zoom_preprocess_factor)
        if self.augment:
            self._augment(svg)
        if self.numericalize:
            svg.numericalize(256)
        return svg


class ArgoverseDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root,
            pad_val=-1,
            batch_size: int = 32,
            fast_run: bool = False,
            train_workers=4,
            val_workers=4,
            test_workers=0,
            augment_train=True,
            augment_val=False,
            augment_test=False,
            **kwargs
    ):
        super(ArgoverseDataModule, self).__init__()
        self.batch_size = batch_size

        self.train_workers = train_workers
        self.val_workers = val_workers
        self.test_workers = test_workers

        if fast_run:
            print("Fast run (train_ds = test_ds = val_ds)")
            self.val_ds = ArgoverseDataset(os.path.join(data_root, f"val"), augment=augment_train, **kwargs)
            self.train_ds = self.test_ds = self.val_ds
        else:
            self.train_ds = ArgoverseDataset(os.path.join(data_root, f"train"), augment=augment_train, **kwargs)
            self.val_ds = ArgoverseDataset(os.path.join(data_root, f"val"), augment=augment_val, **kwargs)
            self.test_ds = ArgoverseDataset(os.path.join(data_root, f"test"), augment=augment_test, **kwargs)

        def collate_fn(batch):
            return pad_collate_fn(batch, pad_val)

        self.collate_fn = collate_fn

        self.train_mean = self.train_ds.get_sequences_mean()
        self.train_mean[:CMDS_CLASSES] = 0.
        self.train_std = self.train_ds.get_sequences_std()
        self.train_std[:CMDS_CLASSES] = 1.

        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.train_workers,
            pin_memory=False,
            persistent_workers=True if self.train_workers > 0 else False,
        )
        self.val_dl = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.val_workers,
            pin_memory=False,
            persistent_workers=True if self.val_workers > 0 else False,
        )

        # Hack in order to fork the dataloader workers before large libraries/models
        # get loaded into memory. Resolves a memory bottleneck with not being able
        # to create a lot of workers (40-60) because of high virtual memory usage
        # (+30GB per worker). Creating (and persisting) the workers before the
        # virtual memory gets allocated seemed to remove the memory problem.
        iter(self.train_dl)
        iter(self.val_dl)

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.test_workers,
            pin_memory=False,
            persistent_workers=True if self.test_workers > 0 else False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.test_workers,
            pin_memory=False,
            persistent_workers=True if self.test_workers > 0 else False,
        )


def main1():
    print("load datasets")
    val_ds_1 = ArgoverseDataset(
        caching_path_prefix="/home/user72/Desktop/argoverse1/val",
        rendered_images_width=512,
        rendered_images_height=512,
        render_on_the_fly=True,
        remove_redundant_features=True,
        numericalize=False,
        augment=False,
    )
    val_ds_2 = ArgoverseDataset(
        caching_path_prefix="/home/user72/Desktop/argoverse2/val",
        rendered_images_width=512,
        rendered_images_height=512,
        render_on_the_fly=True,
        remove_redundant_features=True,
        numericalize=False,
        augment=True,
    )
    print("loaded")

    rendered_images = [val_ds_1[-1][1]]
    rendered_images += [val_ds_2[-1][1] for _ in range(63)]

    grid = torchvision.utils.make_grid(rendered_images)
    plt.imsave(
        f"/home/user72/Desktop/{datetime.now().strftime('%m.%d_%H.%M.%S')}.png",
        grid.permute(1, 2, 0).numpy()
    )
    plt.figure(dpi=300)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
    plt.close()

    print("done")


def main2():
    for ds_name in ["val", "test", "train"]:  # val is the smallest, train the largest subset.
        print(f"Loading subset: {ds_name}")
        ds = ArgoverseDataset(
            caching_path_prefix=f"/home/user72/Desktop/argoverse1/{ds_name}",
            rendered_images_width=64,
            rendered_images_height=64,
            render_on_the_fly=True,
            remove_redundant_features=True,
            numericalize=False,
            augment=False,
        )
        print(f"Loaded the subset: {ds_name}")
        print(f"Number of datapoints: {len(ds._sequences)}")

        def set_unused_to_some_constant(seq, const=5):
            # I put five because I know that my data has a veiwbox of 24, so the xy coordinates
            # are all in the box (0,0)-(24,24).
            return (seq != -1) * seq + (seq == -1) * const

        print(torch.cat([set_unused_to_some_constant(seq).max(0).values[None, :] for seq in ds._sequences]).max(0))
        print(torch.cat([set_unused_to_some_constant(seq).min(0).values[None, :] for seq in ds._sequences]).min(0))

        print(f"Done with {ds}")
        print()
        print()


if __name__ == '__main__':
    main2()
