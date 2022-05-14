import io
import os
import random

import cairosvg
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from deepsvg.svg_dataset import SVGDataset
from torch.utils.data import Dataset, DataLoader

from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.geom import Bbox, Angle, Point
from deepsvg.svglib.svg import SVG
from svglatte.utils.util import pad_collate_fn

CMDS_CLASSES = 7
ARGS_GROUPED_DIM = 13  # 11 + 2


# SEQ_FEATURE_DIM = CMDS_CLASSES + ARGS_GROUPED_DIM

class ArgoverseDataset(Dataset):
    GROUPED_CACHE_FORMAT = "onehot_grouped_commands_concatenated_with_grouped_arguments"  # cached tensor dimensionality: N x (7 + 11)
    SVGTENSOR_DATA_CACHE_FORMAT = "svgtensor_data"  # cached tensor dimensionality: N x (1 + 13)
    SUPPORTED_CACHED_SEQUENCES_FORMAT = {GROUPED_CACHE_FORMAT, SVGTENSOR_DATA_CACHE_FORMAT}

    def __init__(
            self,
            caching_path_prefix,
            cached_sequences_format,
            rendered_images_width=64,
            rendered_images_height=64,
            render_on_the_fly=True,
            remove_redundant_features=True,
            viewbox=24,
            canonicalize_svg=False,
            numericalize=False,
            zoom_preprocess_factor=1.0,  # zoom factor independent of augmentations
            augment=True,
            augment_shear_degrees=20,
            augment_rotate_degrees=180,
            augment_scale_min=0.6,
            augment_scale_max=1.1,
            augment_translate=5.4,
            return_deepsvg_model_input=False,
            deepsvg_model_args=None,
            deepsvg_max_num_groups=None,
            deepsvg_max_seq_len=None,
            deepsvg_max_total_len=None,
            deepsvg_pad_val=None,
    ):
        self.cached_sequences_format = cached_sequences_format
        if self.cached_sequences_format not in ArgoverseDataset.SUPPORTED_CACHED_SEQUENCES_FORMAT:
            raise ValueError(f"Unrecognized (or misconfigured) cache format: {self.cached_sequences_format}."
                             f" Cache format must be one of {ArgoverseDataset.SUPPORTED_CACHED_SEQUENCES_FORMAT}.")

        self.render_on_the_fly = render_on_the_fly
        self.rendered_images_width = rendered_images_width
        self.rendered_images_height = rendered_images_height

        self.zoom_preprocess_factor = zoom_preprocess_factor
        self.remove_redundant_features = remove_redundant_features
        self.canonicalize_svg = canonicalize_svg
        self.numericalize = numericalize
        self.viewbox = viewbox

        self.augment = augment
        self.augment_rotate_degrees = augment_rotate_degrees
        self.augment_translate = augment_translate
        self.augment_scale_min = augment_scale_min
        self.augment_scale_max = augment_scale_max
        self.augment_shear_degrees = augment_shear_degrees

        self.return_deepsvg_model_input = return_deepsvg_model_input
        self.deepsvg_model_args = deepsvg_model_args
        self.deepsvg_max_num_groups = deepsvg_max_num_groups
        self.deepsvg_max_seq_len = deepsvg_max_seq_len
        self.deepsvg_max_total_len = deepsvg_max_total_len
        self.deepsvg_pad_val = deepsvg_pad_val

        self.caching_path_prefix = caching_path_prefix
        self.caching_path_sequences = f"{caching_path_prefix}.sequences.torchsave"
        # self.caching_path_seq_mean = f"{caching_path_prefix}.seq_mean.torchsave"
        # self.caching_path_seq_std = f"{caching_path_prefix}.seq_std.torchsave"
        if not render_on_the_fly:
            self.caching_path_rendered_images = (
                f"{caching_path_prefix}.rendered_images.{rendered_images_width}x{rendered_images_height}.torchsave")

        assert os.path.isfile(self.caching_path_sequences)
        # assert os.path.isfile(self.caching_path_seq_mean)
        # assert os.path.isfile(self.caching_path_seq_std)
        if not render_on_the_fly:
            assert os.path.isfile(self.caching_path_rendered_images)

        self._sequences = torch.load(self.caching_path_sequences)
        self._seq_mean = torch.zeros((self._sequences[0].shape[-1]))  # TODO not cached, thus dummy value
        self._seq_std = torch.zeros((self._sequences[0].shape[-1]))  # TODO not cached, thus dummy value
        # self._seq_mean = torch.load(self.caching_path_seq_mean)
        # self._seq_std = torch.load(self.caching_path_seq_std)
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
            self.sequence_feature_dimenstions = [0, 1, 4, 5, 12, 13, 16 + 2, 17 + 2]
            # Sanity check:
            # torch.cat([s.sum(0)[None, :] for s in self.sequences]).sum(0)
            # tensor([512979., 2584927., 0., 0., 39472., 39472.,
            #         0., -78944., -78944., -78944., -78944., -78944., START_POS_X, START_POS_Y,
            #         -3176850., -3176850., -3176850., -3176850., 28979378., 37385944.])
        else:
            self.sequence_feature_dimenstions = range(CMDS_CLASSES + ARGS_GROUPED_DIM)

    @staticmethod
    def draw_svg(
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
    def svg_to_img(svg, output_width=64, output_height=64):
        pil_image = ArgoverseDataset.draw_svg(svg, output_width, output_height)
        return torch.tensor(np.array(pil_image).transpose(2, 0, 1)[-1:]) / 255.

    def __getitem__(self, idx):
        seq = self._sequences[idx]  # N x 7+11

        # seq (pytorch tensor that was cached) to svgtensor (instance of SVGTensor)
        if self.cached_sequences_format == ArgoverseDataset.GROUPED_CACHE_FORMAT:
            length = torch.tensor(len(seq))
            cache_viewbox_size = 24

            cmds_grouped = torch.argmax(seq[..., :CMDS_CLASSES], dim=-1)
            args_grouped = seq[..., CMDS_CLASSES:]

            svgtensor = SVGTensor.from_cmd_args(cmds_grouped, args_grouped)
            svgtensor.unpad()  # removes eos (and padding, but the preprocessed sequence have no padding)
            svgtensor.drop_sos()

        elif self.cached_sequences_format == ArgoverseDataset.SVGTENSOR_DATA_CACHE_FORMAT:
            length = torch.tensor(len(seq)) + 2  # eos and sos not present
            cache_viewbox_size = 255

            svgtensor_data = seq
            svgtensor = SVGTensor.from_data(svgtensor_data)

        else:
            raise RuntimeError()

        # svgtensor to svg (because svg has the easily accessible augmentation functionality (zoom, translate, ...)
        svg = SVG.from_tensor(svgtensor.data, viewbox=Bbox(cache_viewbox_size))

        # working in the svg domain (allows for simpler preprocessing and rendering compared to seq or svgtensor)
        svg = self._preprocess(svg)
        if self.render_on_the_fly:
            rendered_image = self._render_on_the_fly(idx, svg)
        else:
            rendered_image = self._rendered_images[idx]

        if self.return_deepsvg_model_input:
            model_inputs = SVGDataset.tensors_to_model_inputs(
                tensors_separately=svg.split_paths().to_tensor(concat_groups=False, PAD_VAL=self.deepsvg_pad_val),
                fillings=svg.to_fillings(),

                max_num_groups=self.deepsvg_max_num_groups,
                max_seq_len=self.deepsvg_max_seq_len,
                max_total_len=self.deepsvg_max_total_len,
                pad_val=self.deepsvg_pad_val,
                model_args=self.deepsvg_model_args,
                label=None,
            )
            return model_inputs, rendered_image, length

        # svg back to svgtensor
        tensor_data = svg.to_tensor(concat_groups=True, PAD_VAL=-1)
        svgtensor = SVGTensor.from_data(tensor_data)

        # svgtensor back to seq
        svgtensor.add_eos()
        svgtensor.add_sos()
        cmds_grouped = svgtensor.cmds()
        cmds_grouped_onehot = torch.nn.functional.one_hot(cmds_grouped.to(torch.int64), num_classes=CMDS_CLASSES)
        args_grouped = svgtensor.args(with_start_pos=True)
        seq = torch.cat((cmds_grouped_onehot, args_grouped), dim=1)  # N x 7+13!
        debug_length = torch.tensor(len(seq))
        assert length == debug_length
        assert CMDS_CLASSES + ARGS_GROUPED_DIM == len(seq[0])

        seq = seq[:, self.sequence_feature_dimenstions]
        return seq, rendered_image, length

    def __len__(self):
        return len(self._sequences)

    def get_number_of_sequence_dimensions(self):
        return len(self.sequence_feature_dimenstions)

    def _render_on_the_fly(self, idx, svg):
        rendered_image = ArgoverseDataset.svg_to_img(
            svg,
            self.rendered_images_width,
            self.rendered_images_height
        )
        return rendered_image

    def get_sequences_mean(self):
        return torch.tensor([0])
        return self._seq_mean[self.sequence_feature_dimenstions]

    def get_sequences_std(self):
        return torch.tensor([1])
        return self._seq_std[self.sequence_feature_dimenstions]

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
        svg.normalize(Bbox(self.viewbox))
        if self.canonicalize_svg:
            svg.canonicalize()
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

        self.collate_fn = collate_fn if not kwargs['return_deepsvg_model_input'] else None

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
