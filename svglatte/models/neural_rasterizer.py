import argparse
from collections import OrderedDict
from datetime import datetime

import pytorch_lightning as pl
import torch
import torchvision
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F

from svglatte.dataset import DeepVecFontDataModule
from svglatte.dataset import argoverse_dataset
from svglatte.dataset import deepsvg_dataset, deepvecfont_dataset
from svglatte.models.custom_lstms import SequenceEncoder
from svglatte.models.lstm_layernorm import LayerNormLSTMEncoder
from svglatte.models.vgg_contextual_loss import VGGContextualLoss
from svglatte.utils.util import AttrDict, nice_print, HORSE


def get_parser_main_model():
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--experiment_name', type=str, default='svglatte')
    parser.add_argument('--experiment_version', type=str, default=None)
    parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--gpus', type=int, default=-1)

    # optimizer
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam optimizer')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')

    # loss weight
    parser.add_argument('--kl_beta', type=float, default=0.01, help='latent code kl loss beta')
    parser.add_argument('--pt_c_loss_w', type=float, default=0.001, help='the weight of perceptual content loss')
    parser.add_argument('--cx_loss_w', type=float, default=0.1, help='the weight of contextual loss')
    parser.add_argument('--l1_loss_w', type=float, default=1, help='the weight of image reconstruction l1 loss')
    parser.add_argument('--mdn_loss_w', type=float, default=1.0, help='the weight of mdn loss')
    parser.add_argument('--softmax_loss_w', type=float, default=1.0, help='the weight of softmax ce loss')

    # encoder
    parser.add_argument('--encoder_type', type=str, choices=[
        'dvf_lstm',
        'fc',
        'lstm', 'fc_lstm_original', 'fc_lstm', 'residual_lstm', 'lstm+mha',
    ])
    parser.add_argument('--latte_ingredients', type=str, default='hc', choices=['h', 'c', 'hc'])
    # parser.add_argument('--lstm_input_size', type=int, default=512, help='lstm encoder input_size')
    parser.add_argument('--lstm_hidden_size', type=int, default=512, help='lstm encoder hidden_size')
    parser.add_argument('--lstm_num_layers', type=int, default=4, help='svg encoder number of hidden layers')
    parser.add_argument('--lstm_dropout', type=float, default=0.0, help='lstm dropout')
    parser.add_argument('--lstm_bidirectional', action='store_true', help='')
    parser.add_argument('--no_sequence_packing', action='store_true', help='')
    parser.add_argument('--mha_num_layers', type=int, default=4, help='')
    parser.add_argument('--mha_hidden_size', type=int, default=512, help='')
    parser.add_argument('--mha_num_heads', type=int, default=8, help='')
    parser.add_argument('--mha_dropout', type=float, default=0.0, help='')
    parser.add_argument('--dvf_lstm_unpack_output', action='store_true', help='')

    # datasets
    parser.add_argument('--cache_to_disk', action='store_false', help='')
    parser.add_argument('--clean_disk_cache', action='store_true', help='')
    parser.add_argument('--dataset', type=str, default='deepvecfont', choices=['deepvecfont', 'deepsvg', 'argoverse'])
    parser.add_argument('--standardize_input_sequences', action='store_true', help='')
    ## deepvecfont
    parser.add_argument('--deepvecfont_data_root', type=str, default='data/vecfont_dataset')
    parser.add_argument('--deepvecfont_max_seq_len', type=int, default=51, help='maximum length of sequence')
    parser.add_argument('--deepvecfont_seq_feature_dim', type=int, default=10,
                        help='feature dim (like vocab size) of one step of sequence feature')
    parser.add_argument('--deepvecfont_glyphs_in_alphabet', type=int, default=52, help='number of glyphs in a alphabet')
    ## deepsvg
    parser.add_argument('--deepsvg_data_root', type=str, default='data/deepsvg_dataset/icons_tensor/')
    parser.add_argument('--deepsvg_meta_filepath', type=str, default='data/deepsvg_dataset/icons_meta.csv')
    parser.add_argument('--deepsvg_max_num_groups', type=int, default=120, help='maximum number of groups')
    parser.add_argument('--deepsvg_max_seq_len', type=int, default=200, help='maximum length of sequence')
    parser.add_argument('--deepsvg_max_total_len', type=int, default=2000, help='maximum total length of an svg')
    ## argoverse
    parser.add_argument('--argoverse_data_root', type=str, default='data/argoverse/')
    parser.add_argument('--argoverse_rendered_images_width', type=int, default=64, help='Height of rendered images')
    parser.add_argument('--argoverse_rendered_images_height', type=int, default=64, help='Width of rendered images')
    parser.add_argument('--argoverse_fast_run', action='store_true',
                        help='To gave a faster run, we use the smallest dataset subset for all datasets')
    parser.add_argument('--argoverse_render_onthefly', action='store_true',
                        help='Render images of any size on-the-fly, i.e. do not used the cached images')
    parser.add_argument('--argoverse_keep_redundant_features', action='store_true',
                        help='Keep the redundant features in svg tensor sequences. These features are redundant '
                             'because (1.) DeepSVG svg tensors have a few unused features, we keep them internally to'
                             'be able to use the DeepSVG library, (2.) we leave only lines when preprocessing'
                             'Argoverse (curves are simplfied into lines).')
    parser.add_argument('--argoverse_train_workers', type=int, default=4, help='')
    parser.add_argument('--argoverse_val_workers', type=int, default=4, help='')
    parser.add_argument('--argoverse_test_workers', type=int, default=0, help='')
    return parser


# Adapted from:
# https://github.com/yizhiwang96/deepvecfont/blob/3ba4adb0406f16a6f387c5e12dd12286c9c341e8/models/neural_rasterizer.py
class NeuralRasterizer(pl.LightningModule):
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            optimizer_args,
            l1_loss_w,
            cx_loss_w,
            standardize_input_sequences,
            dataset_name=None,
            train_mean=None,
            train_std=None,
    ):
        super(NeuralRasterizer, self).__init__()
        self.optimizer_args = optimizer_args
        self.dataset_name = dataset_name

        # sequence encoder
        self.encoder = encoder
        self.standardize_input_sequences = standardize_input_sequences
        self.train_mean = train_mean
        self.train_std = train_std

        # image decoder
        self.decoder = decoder

        # vgg contextual loss
        self.l1_loss_w = l1_loss_w
        self.cx_loss_w = cx_loss_w
        self.vggcxlossfunc = VGGContextualLoss()

    # TODO what does this method do
    # def init_state_input(self, sampled_bottleneck):
    #     init_state_hidden = []
    #     init_state_cell = []
    #     for i in range(self.num_hidden_layers):
    #         unbottleneck = self.unbottlenecks[i](sampled_bottleneck)
    #         (h0, c0) = unbottleneck[:, :self.unbottleneck_dim // 2], unbottleneck[:, self.unbottleneck_dim // 2:]
    #         init_state_hidden.append(h0.unsqueeze(0))
    #         init_state_cell.append(c0.unsqueeze(0))
    #     init_state_hidden = torch.cat(init_state_hidden, dim=0)
    #     init_state_cell = torch.cat(init_state_cell, dim=0)
    #     init_state = {}
    #     init_state['hidden'] = init_state_hidden
    #     init_state['cell'] = init_state_cell
    #     return init_state

    def forward(self, trg_seq_padded, lengths):
        latte = self.encoder(trg_seq_padded, lengths)
        return latte

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.optimizer_args.lr, betas=(self.optimizer_args.beta1, self.optimizer_args.beta2),
            eps=self.optimizer_args.eps, weight_decay=self.optimizer_args.weight_decay
        )
        return optimizer

    def training_step(self, train_batch, batch_idx, subset="train"):
        trg_seq_padded, trg_img, lengths = train_batch
        if self.standardize_input_sequences:
            trg_seq_padded = (trg_seq_padded - self.train_mean.to(self.device)) / self.train_std.to(self.device)

        # get latent
        latte = self.forward(trg_seq_padded, lengths.cpu())

        # decode to get the raster
        dec_input = latte
        dec_input = dec_input.unsqueeze(-1).unsqueeze(-1)
        dec_out = self.decoder(dec_input)

        l1_loss = F.l1_loss(dec_out, trg_img)

        # compute losses
        vggcx_loss = self.vggcxlossfunc(dec_out, trg_img)
        # vggcx_loss = {'cx_loss': 0.0}
        loss = self.l1_loss_w * l1_loss + self.cx_loss_w * vggcx_loss['cx_loss']

        # results
        results = {
            "loss": loss,
            "gen_imgs": dec_out,
            "img_l1_loss": l1_loss,
            "img_vggcx_loss": vggcx_loss["cx_loss"],
        }

        # logging
        self.log(f'Loss/{subset}/loss', loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'Loss/{subset}/img_l1_loss', self.l1_loss_w * l1_loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'Loss/{subset}/img_perceptual_loss', self.cx_loss_w * vggcx_loss['cx_loss'],
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 200 == 0:
            zipped = torch.concat(list(map(torch.cat, zip(dec_out[:32, None], trg_img[:32, None]))))
            zipped_image = torchvision.utils.make_grid(zipped)
            for logger in self.loggers:
                if type(logger) == WandbLogger:
                    logger.experiment.log({
                        f'Images/{subset}/{batch_idx}-output_img': [wandb.Image(
                            zipped_image,
                            caption=f"loss:{loss} img_l1_loss:{l1_loss} img_vggcx_loss:{vggcx_loss['cx_loss']}")
                        ]})
                elif type(logger) == TensorBoardLogger:
                    logger.experiment.add_image(
                        f'Images/{subset}/{batch_idx}-output_img',
                        zipped_image,
                        self.current_epoch
                    )

            if self.global_step == 0 and self.dataset_name == "deepvecfont":
                for i, seq in enumerate(trg_seq_padded[:32]):
                    seq = (seq * self.train_std.to(self.device) + self.train_mean.to(self.device)).cpu().numpy()
                    from svglatte.dataset.deepvecfont_svg_utils import render
                    tgt_svg = render(seq)
                    for logger in self.loggers:
                        if type(logger) == WandbLogger:
                            logger.experiment.log({f"Svg/{subset}/{batch_idx}-{i}-target_svg": wandb.Html(tgt_svg)})
                        elif type(logger) == TensorBoardLogger:
                            logger.experiment.add_text(
                                f"Svg/{subset}/{batch_idx}-{i}-target_svg",
                                tgt_svg,
                                global_step=self.current_epoch
                            )

        # return results
        return results

    def validation_step(self, val_batch, batch_idx):
        results = self.training_step(val_batch, batch_idx, subset="val")
        return results

    def test_step(self, test_batch, batch_idx):
        results = self.training_step(test_batch, batch_idx, subset="test")
        return results


def get_dataset(config):
    if config.dataset == "deepvecfont":
        seq_feature_dim = deepvecfont_dataset.SEQ_FEATURE_DIM
        dm = DeepVecFontDataModule(
            data_root=config.deepvecfont_data_root,
            max_seq_len=config.deepvecfont_max_seq_len,
            seq_feature_dim=config.deepvecfont_seq_feature_dim,
            glyphs_in_alphabet=config.deepvecfont_glyphs_in_alphabet,
            batch_size=config.batch_size,
            cache_to_disk=config.cache_to_disk,
            clean_disk_cache=config.clean_disk_cache,
        )
    elif config.dataset == "deepsvg":
        seq_feature_dim = deepsvg_dataset.SEQ_FEATURE_DIM
        dm = deepsvg_dataset.DeepSVGDataModule(
            data_root=config.deepsvg_data_root,
            meta_filepath=config.deepsvg_meta_filepath,
            max_num_groups=config.deepsvg_max_num_groups,
            max_seq_len=config.deepsvg_max_seq_len,
            max_total_len=config.deepsvg_max_total_len,
            batch_size=config.batch_size,
            cache_to_disk=config.cache_to_disk,
            clean_disk_cache=config.clean_disk_cache,
        )
    elif config.dataset == "argoverse":
        seq_feature_dim = argoverse_dataset.SEQ_FEATURE_DIM
        dm = argoverse_dataset.ArgoverseDataModule(
            data_root=config.argoverse_data_root,
            rendered_images_width=config.argoverse_rendered_images_width,
            rendered_images_height=config.argoverse_rendered_images_height,
            batch_size=config.batch_size,
            render_on_the_fly=config.argoverse_render_onthefly,
            remove_redundant_features=not config.argoverse_keep_redundant_features,
            fast_run=config.argoverse_fast_run,
            train_workers=config.argoverse_train_workers,
            val_workers=config.argoverse_val_workers,
            test_workers=config.argoverse_test_workers,
        )
        seq_feature_dim = dm.train_ds.get_number_of_sequence_dimensions()
    else:
        raise Exception(f"Invalid dataset passed: {config.dataset}")
    return dm, seq_feature_dim


class FCEncoder(nn.Module):

    def __init__(
            self,
            neurons_per_layer,
            activation_module,
            use_batchnorm=True,
            **kwargs
    ):
        super(FCEncoder, self).__init__()
        self.neurons_per_layer = neurons_per_layer

        layers = OrderedDict()
        for i in range(1, len(self.neurons_per_layer)):
            layers[f"ll_{i}"] = nn.Linear(
                in_features=self.neurons_per_layer[i - 1],
                out_features=self.neurons_per_layer[i],
                bias=not use_batchnorm
            )

            if i != len(self.dims) - 1:
                layers[f"bn_{i}"] = nn.BatchNorm1d(self.neurons_per_layer[i])
                layers[f"a_{i}"] = activation_module()

        self.seq = nn.Sequential(layers)

    def forward(self, input_sequences, _):
        # TODO how to fix the input length?
        return self.seq(input_sequences.reshape(input_sequences.shape[0], -1))


def get_encoder(config):
    # TODO refactor to use hydra or subparsers
    encoder_args = AttrDict()
    encoder_args.update({
        "encoder_type": config.encoder_type,
        "latte_ingredients": config.latte_ingredients,
        "lstm_input_size": config.lstm_input_size,
        "lstm_hidden_size": config.lstm_hidden_size,
        "lstm_num_layers": config.lstm_num_layers,
        "lstm_dropout": config.lstm_dropout,
        "lstm_bidirectional": config.lstm_bidirectional,
        "pack_sequences": not config.no_sequence_packing,
        "mha_num_layers": config.mha_num_layers,
        "mha_hidden_size": config.mha_hidden_size,
        "mha_num_heads": config.mha_num_heads,
        "mha_dropout": config.mha_dropout,
    })
    if config.encoder_type == "dvf_lstm":
        encoder_args.unpack_output = config.dvf_lstm_unpack_output
        encoder = LayerNormLSTMEncoder(**encoder_args)
    elif config.encoder_type == "fc":
        raise NotImplementedError(
            "The FC baseline is not yet implemnted. Need to figure out how to hae fixed input size.")
        encoder = FCEncoder(**encoder_args)
    else:
        encoder = SequenceEncoder(**encoder_args)

    return encoder


class Decoder(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            ngf=16,  # 64
            norm_layer=nn.LayerNorm,
            kernel_size_list=(3, 3, 5, 5, 5, 5),  # 5
            stride_list=(2, 2, 2, 2, 2, 2),  # 3
            padding_list=(1, 1, 2, 2, 2, 2),
            output_padding_list=(1, 1, 1, 1, 1, 1),
    ):
        super(Decoder, self).__init__()
        decoder = []
        assert len(kernel_size_list) == len(stride_list) == len(padding_list) == len(output_padding_list)
        image_sizes = []
        image_size = 1
        for k, s, p, op in zip(kernel_size_list, stride_list, padding_list, output_padding_list):
            image_size = (image_size - 1) * s - 2 * p + 1 * (k - 1) + op + 1
            image_sizes.append(image_size)
        n_upsampling = len(kernel_size_list)
        mult = 2 ** n_upsampling

        conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=int(ngf * mult / 2),
            kernel_size=kernel_size_list[0],
            stride=stride_list[0],
            padding=padding_list[0],
            output_padding=output_padding_list[0]
        )
        decoder += [conv, norm_layer([int(ngf * mult / 2), image_sizes[0], image_sizes[0]]), nn.ReLU(True)]
        for i in range(1, n_upsampling):
            mult = 2 ** (n_upsampling - i)
            conv = nn.ConvTranspose2d(
                in_channels=ngf * mult,
                out_channels=int(ngf * mult / 2),
                kernel_size=kernel_size_list[i],
                stride=stride_list[i],
                padding=padding_list[i],
                output_padding=output_padding_list[i]
            )
            decoder += [conv, norm_layer([int(ngf * mult / 2), image_sizes[i], image_sizes[i]]), nn.ReLU(True)]
        decoder += [nn.Conv2d(ngf, out_channels, kernel_size=7, padding=7 // 2)]
        decoder += [nn.Sigmoid()]
        self.decode = nn.Sequential(*decoder)

    def forward(self, latte):
        return self.decode(latte)


def main(config):
    pl.seed_everything(72)

    dm, seq_feature_dim = get_dataset(config)
    config.lstm_input_size = seq_feature_dim

    encoder = get_encoder(config)
    if config.argoverse_rendered_images_width == 64 and config.argoverse_rendered_images_height == 64:
        decoder = Decoder(
            in_channels=encoder.output_size,
            out_channels=1
        )
    elif config.argoverse_rendered_images_width == 200 and config.argoverse_rendered_images_height == 200:
        decoder = Decoder(
            in_channels=encoder.output_size,
            out_channels=1,
            kernel_size_list=(3, 3, 3, 3, 3, 3, 5, 5),
            stride_list=(2, 2, 2, 2, 2, 2, 2, 2),
            padding_list=(1, 1, 1, 2, 2, 2, 2, 2),
            output_padding_list=(1, 1, 1, 1, 1, 1, 1, 1),
            ngf=4,  # 64
            # norm_layer=nn.LayerNorm,
        )
    else:
        raise Exception(f"Image size {config.argoverse_rendered_images_width}x{config.argoverse_rendered_images_height}"
                        f" not supported")

    adam_optimizer_args = AttrDict()
    adam_optimizer_args.update({
        "lr": config.lr,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "eps": config.eps,
        "weight_decay": config.weight_decay
    })
    neural_rasterizer = NeuralRasterizer(
        encoder=encoder,
        decoder=decoder,
        optimizer_args=adam_optimizer_args,
        l1_loss_w=config.l1_loss_w,
        cx_loss_w=config.cx_loss_w,
        dataset_name=config.dataset,
        train_mean=dm.train_mean,
        train_std=dm.train_std,
        standardize_input_sequences=config.standardize_input_sequences,
    )
    print(neural_rasterizer)

    if config.experiment_version is None:
        config.experiment_version = f"{config.dataset[:2]}" \
                                    f"_{config.encoder_type}" \
                                    f"{'_bi' if config.lstm_bidirectional else ''}" \
                                    f"_h={config.lstm_hidden_size}" \
                                    f"_i-{config.latte_ingredients}" \
                                    f"_l={config.lstm_num_layers}" \
                                    f"{'_nops' if config.no_sequence_packing else ''}" \
                                    f"_e={config.n_epochs}" \
                                    f"_b={config.batch_size}" \
                                    f"_lr={config.lr}" \
                                    f"_wd={config.weight_decay}" \
                                    f"_{datetime.now().strftime('%m.%d_%H.%M.%S')}"
    wandb_logger = WandbLogger(project=config.experiment_name, version=config.experiment_version.replace("=", "-"))
    wandb_logger.watch(neural_rasterizer)
    tb_logger = TensorBoardLogger("logs", name=config.experiment_name, version=config.experiment_version, )
    csv_logger = CSVLogger("logs", name=config.experiment_name, version=config.experiment_version, )

    if torch.cuda.is_available() and config.gpus != 0:
        trainer = Trainer(
            max_epochs=config.n_epochs,
            default_root_dir="logs",
            logger=[wandb_logger, tb_logger, csv_logger],
            callbacks=[
                EarlyStopping(monitor="Loss/val/loss", mode="min", patience=100, check_on_train_epoch_end=False),
                ModelCheckpoint(monitor="Loss/val/loss", save_last=True),
            ],
            gpus=config.gpus,
            accelerator="gpu",
            strategy='dp',
            # strategy=DDPStrategy(find_unused_parameters=False),
            # resume_from_checkpoint="logs/svglatte_svglatte__2022.04.09_16.04.39/3ireo6s9_0/checkpoints/epoch=161-step=211572.ckpt",
            # num_sanity_val_steps=0,
            # fast_dev_run=True,
            # limit_train_batches=10,
            # limit_val_batches=10,
            # detect_anomaly=True,
            # profiler="simple",
        )
    else:
        trainer = Trainer(
            max_epochs=config.n_epochs,
            default_root_dir="logs",
            logger=[wandb_logger, tb_logger, csv_logger],
            callbacks=[
                EarlyStopping(monitor="Loss/val/loss", mode="min", patience=100, check_on_train_epoch_end=False),
                ModelCheckpoint(monitor="Loss/val/loss", save_last=True),
            ],
        )
    trainer.fit(neural_rasterizer, dm)
    trainer.test(neural_rasterizer, dm, ckpt_path='best')


if __name__ == "__main__":
    nice_print(HORSE)
    parser = get_parser_main_model()
    args = parser.parse_args()
    print(args)
    main(args)
