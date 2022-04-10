import argparse
import os
import pickle

import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as T
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

from data_utils.svg_utils import render
from svglatte.models.lstm_layernorm import LayerNormLSTM
from svglatte.models.vgg_contextual_loss import VGGContextualLoss
from svglatte.utils.util import get_str_formatted_time

HORSE = """               .,,.
             ,;;*;;;;,
            .-'``;-');;.
           /'  .-.  /*;;
         .'    \\d    \\;;               .;;;,
        / o      `    \\;    ,__.     ,;*;;;*;,
        \\__, _.__,'   \\_.-') __)--.;;;;;*;;;;,
         `""`;;;\\       /-')_) __)  `\' ';;;;;;
            ;*;;;        -') `)_)  |\\ |  ;;;;*;
            ;;;;|        `---`    O | | ;;*;;;
            *;*;\\|                 O  / ;;;;;*
           ;;;;;/|    .-------\\      / ;*;;;;;
          ;;;*;/ \\    |        '.   (`. ;;;*;;;
          ;;;;;'. ;   |          )   \\ | ;;;;;;
          ,;*;;;;\\/   |.        /   /` | ';;;*;
           ;latte/    |/       /   /__/   ';;;
           '*;;;/     |       /    |      ;*;
                `""""`        `""""`     ;'"""


def nice_print(msg, last=False):
    print()
    print("\033[0;35m" + msg + "\033[0m")
    if last:
        print()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_parser_main_model():
    parser = argparse.ArgumentParser()
    # TODO: basic parameters training related
    parser.add_argument('--model_name', type=str, default='main_model', choices=['main_model', 'neural_raster'],
                        help='current model_name')
    parser.add_argument('--bottleneck_bits', type=int, default=128, help='latent code number of bottleneck bits')
    parser.add_argument('--glyphs_in_alphabet', type=int, default=52, help='number of glyphs in a alphabet')
    parser.add_argument('--ref_nshot', type=int, default=4, help='reference number')
    parser.add_argument('--in_channel', type=int, default=1, help='input image channel')
    parser.add_argument('--out_channel', type=int, default=1, help='output image channel')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--image_size', type=int, default=64, help='image size')
    parser.add_argument('--max_seq_len', type=int, default=51, help='maximum length of sequence')
    parser.add_argument('--seq_feature_dim', type=int, default=10,
                        help='feature dim (like vocab size) of one step of sequence feature')
    # experiment related
    parser.add_argument('--init_epoch', type=int, default=0, help='init epoch')
    parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--experiment_name', type=str, default='svglatte')
    parser.add_argument('--data_root', type=str, default='data/vecfont_dataset')
    parser.add_argument('--ckpt_freq', type=int, default=25, help='save checkpoint frequency of epoch')
    parser.add_argument('--sample_freq', type=int, default=200, help='sample train output of steps')
    parser.add_argument('--val_freq', type=int, default=1000, help='sample validate output of steps')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam optimizer')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--tboard', type=bool, default=True, help='whether use tensorboard to visulize loss')
    parser.add_argument('--test_sample_times', type=int, default=20, help='the sample times when testing')
    # loss weight
    parser.add_argument('--kl_beta', type=float, default=0.01, help='latent code kl loss beta')
    parser.add_argument('--pt_c_loss_w', type=float, default=0.001, help='the weight of perceptual content loss')
    parser.add_argument('--cx_loss_w', type=float, default=0.1, help='the weight of contextual loss')
    parser.add_argument('--l1_loss_w', type=float, default=1, help='the weight of image reconstruction l1 loss')
    parser.add_argument('--mdn_loss_w', type=float, default=1.0, help='the weight of mdn loss')
    parser.add_argument('--softmax_loss_w', type=float, default=1.0, help='the weight of softmax ce loss')
    # neural rasterizer
    parser.add_argument('--use_nr', type=bool, default=True, help='whether to use neural rasterization during training')
    # LSTM related
    parser.add_argument('--hidden_size', type=int, default=512, help='lstm encoder hidden_size')
    parser.add_argument('--num_hidden_layers', type=int, default=4, help='svg decoder number of hidden layers')
    parser.add_argument('--rec_dropout', type=float, default=0.3, help='LSTM rec dropout')
    parser.add_argument('--ff_dropout', type=float, default=0.5, help='LSTM feed forward dropout')
    # MDN related
    parser.add_argument('--num_mixture', type=int, default=50, help='')
    parser.add_argument('--mix_temperature', type=float, default=0.00001, help='')
    parser.add_argument('--gauss_temperature', type=float, default=0.00001, help='')
    # parser.add_argument('--mix_temperature', type=float, default=0.0001, help='')
    # parser.add_argument('--gauss_temperature', type=float, default=0.01, help='')
    parser.add_argument('--dont_reduce_loss', type=bool, default=False, help='')
    # testing related
    parser.add_argument('--test_epoch', type=int, default=125, help='the testing checkpoint')
    parser.add_argument('--test_fontid', type=int, default=0, help='the testing font id')

    parser.add_argument('--cache_to_disk', action='store_false', help='')
    parser.add_argument('--clean_disk_cache', action='store_true', help='')
    return parser


# Adapted from:
# https://github.com/yizhiwang96/deepvecfont/blob/3ba4adb0406f16a6f387c5e12dd12286c9c341e8/models/neural_rasterizer.py
class NeuralRasterizer(pl.LightningModule):
    def __init__(
            self,
            optimizer_args,
            feature_dim,
            hidden_size,
            num_hidden_layers,
            ff_dropout_p,
            rec_dropout_p,
            input_nc,
            output_nc,
            l1_loss_w,
            cx_loss_w,
            ngf=64,
            bottleneck_bits=32,
            norm_layer=nn.LayerNorm,
            mode='train',
            decoder_kernel_size_list=(3, 3, 5, 5, 5, 5),
            decoder_stride_list=(2, 2, 2, 2, 2, 2),
            train_mean=None,
            train_std=None,
    ):
        """
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(NeuralRasterizer, self).__init__()

        self.train_mean = train_mean
        self.train_std = train_std

        self.optimizer_args = optimizer_args
        self.l1_loss_w = l1_loss_w
        self.cx_loss_w = cx_loss_w

        # seq encoder
        self.input_dim = feature_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

        self.bottleneck_bits = bottleneck_bits
        self.unbottleneck_dim = self.hidden_size * 2

        # TODO these lines seem not to be needed
        # self.ff_dropout_p = float(mode == 'train') * ff_dropout_p
        # self.rec_dropout_p = float(mode == 'train') * rec_dropout_p
        # self.pre_lstm_fc = nn.Linear(self.input_dim, self.hidden_size)
        self.lstm = LayerNormLSTM(self.input_dim, self.hidden_size, self.num_hidden_layers)

        # image decoder
        decoder = []
        assert len(decoder_kernel_size_list) == len(decoder_stride_list)
        n_upsampling = len(decoder_kernel_size_list)
        mult = 2 ** n_upsampling

        conv = nn.ConvTranspose2d(
            in_channels=input_nc,
            out_channels=int(ngf * mult / 2),
            kernel_size=decoder_kernel_size_list[0],
            stride=decoder_stride_list[0],
            padding=decoder_kernel_size_list[0] // 2,
            output_padding=decoder_stride_list[0] - 1
        )
        decoder += [conv, norm_layer([int(ngf * mult / 2), 2, 2]), nn.ReLU(True)]
        for i in range(1, n_upsampling):
            mult = 2 ** (n_upsampling - i)
            conv = nn.ConvTranspose2d(
                in_channels=ngf * mult,
                out_channels=int(ngf * mult / 2),
                kernel_size=decoder_kernel_size_list[i],
                stride=decoder_stride_list[i],
                padding=decoder_kernel_size_list[i] // 2,
                output_padding=decoder_stride_list[i] - 1
            )
            decoder += [conv, norm_layer([int(ngf * mult / 2), 2 ** (i + 1), 2 ** (i + 1)]), nn.ReLU(True)]
        decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=7 // 2)]
        decoder += [nn.Sigmoid()]
        self.decode = nn.Sequential(*decoder)

        # vgg contextual loss
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

    def forward(self, trg_seq):
        output, (hidden, cell) = self.lstm(trg_seq.transpose(0, 1), None)
        latte = torch.cat((cell[-1, :, :], hidden[-1, :, :]), -1)
        return latte

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.optimizer_args.lr, betas=(self.optimizer_args.beta1, self.optimizer_args.beta2),
            eps=self.optimizer_args.eps, weight_decay=self.optimizer_args.weight_decay
        )
        return optimizer

    def training_step(self, train_batch, batch_idx, subset="train"):
        trg_seq, trg_img = train_batch
        trg_seq = (trg_seq - self.train_mean.to(self.device)) / self.train_std.to(self.device)

        # get latent
        latte = self.forward(trg_seq)

        # decode to get the raster
        dec_input = latte
        # dec_input = dec_input.view(dec_input.size(0), dec_input.size(1), 1, 1)
        dec_input = dec_input.unsqueeze(-1).unsqueeze(-1)
        dec_out = self.decode(dec_input)

        l1_loss = F.l1_loss(dec_out, trg_img)

        # compute losses
        vggcx_loss = self.vggcxlossfunc(dec_out, trg_img)
        loss = self.l1_loss_w * l1_loss + self.cx_loss_w * vggcx_loss['cx_loss']

        # results
        results = {
            "loss": loss,
            "gen_imgs": dec_out,
            "img_l1_loss": l1_loss,
            "img_vggcx_loss": vggcx_loss["cx_loss"],
        }

        # logging
        # loggers[0] --> wandb
        # loggers[1] --> tb
        self.log(f'Loss/{subset}/loss', loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'Loss/{subset}/img_l1_loss', self.l1_loss_w * l1_loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'Loss/{subset}/img_perceptual_loss', self.cx_loss_w * vggcx_loss['cx_loss'],
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # if self.global_step % 50 == 0:
        #     self.loggers[0].log_metrics({
        #         f'Loss/{subset}/loss': loss,
        #         f'Loss/{subset}/img_l1_loss': self.l1_loss_w * l1_loss,
        #         f'Loss/{subset}/img_perceptual_loss': self.cx_loss_w * vggcx_loss['cx_loss'],
        #     })
        #     self.loggers[1].experiment.add_scalar(f'Loss/{subset}/loss', loss, self.global_step)
        #     self.loggers[1].experiment.add_scalar(f'Loss/{subset}/img_l1_loss', self.l1_loss_w * l1_loss,
        #                                           self.global_step)
        #     self.loggers[1].experiment.add_scalar(f'Loss/{subset}/img_perceptual_loss',
        #                                           self.cx_loss_w * vggcx_loss['cx_loss'],
        #                                           self.global_step)

        if batch_idx % 200 == 0:
            zipped = torch.concat(list(map(torch.cat, zip(dec_out[:32, None], trg_img[:32, None]))))
            zipped_image = torchvision.utils.make_grid(zipped)
            self.loggers[0].experiment.log({
                f'Images/{subset}/{batch_idx}-output_img': [wandb.Image(
                    zipped_image, caption=f"loss:{loss} img_l1_loss:{l1_loss} img_vggcx_loss:{vggcx_loss['cx_loss']}")
                ]})
            self.loggers[1].experiment.add_image(
                f'Images/{subset}/{batch_idx}-output_img',
                zipped_image,
                self.current_epoch
            )

            if self.global_step == 0:
                for i, seq in enumerate(trg_seq[:32]):
                    seq = (seq * self.train_std.to(self.device) + self.train_mean.to(self.device)).cpu().numpy()
                    tgt_svg = render(seq)
                    self.loggers[0].experiment.log({f"Svg/{subset}/{batch_idx}-{i}-target_svg": wandb.Html(tgt_svg)})
                    self.loggers[1].experiment.add_text(
                        f"Svg/{subset}/{batch_idx}-{i}-target_svg",
                        tgt_svg,
                        global_step=self.current_epoch
                    )

        # return results
        return results

    def validation_step(self, val_batch, batch_idx):
        results = self.training_step(val_batch, batch_idx, subset="val")
        return results


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


def main(config):
    pl.seed_everything(72)

    dm = DeepVecFontDataModule(
        data_root=config.data_root,
        max_seq_len=config.max_seq_len,
        seq_feature_dim=config.seq_feature_dim,
        glyphs_in_alphabet=config.glyphs_in_alphabet,
        batch_size=config.batch_size,
        cache_to_disk=config.cache_to_disk,
        clean_disk_cache=config.clean_disk_cache,
    )

    adam_optimizer_args = AttrDict()
    adam_optimizer_args.update({
        "lr": config.lr,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "eps": config.eps,
        "weight_decay": config.weight_decay
    })
    neural_rasterizer = NeuralRasterizer(
        optimizer_args=adam_optimizer_args,
        l1_loss_w=config.l1_loss_w,
        cx_loss_w=config.cx_loss_w,
        feature_dim=config.seq_feature_dim,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        ff_dropout_p=config.ff_dropout,
        rec_dropout_p=config.rec_dropout,
        input_nc=2 * config.hidden_size,
        output_nc=1,
        ngf=16,
        bottleneck_bits=config.bottleneck_bits,
        norm_layer=nn.LayerNorm,
        mode='train',
        train_mean=dm.train_mean,
        train_std=dm.train_std,
    )

    wandb_logger = WandbLogger(project=config.experiment_name, log_model="all")
    wandb_logger.watch(neural_rasterizer)

    experiment_name = f"{config.experiment_name}__{get_str_formatted_time()}"
    tb_logger = TensorBoardLogger("logs", name=experiment_name)

    if torch.cuda.is_available():
        trainer = Trainer(
            max_epochs=config.n_epochs,
            default_root_dir="logs",
            logger=[wandb_logger, tb_logger],
            callbacks=[
                EarlyStopping(monitor="Loss/val/loss", mode="min", patience=100, check_on_train_epoch_end=False)],
            gpus=config.gpus,
            accelerator="gpu",
            strategy='dp',
            # strategy=DDPStrategy(find_unused_parameters=False),
            resume_from_checkpoint="logs/svglatte_svglatte__2022.04.09_16.04.39/3ireo6s9_0/checkpoints/epoch=161-step=211572.ckpt",
            # fast_dev_run=True,
            # limit_train_batches=10,
            # limit_val_batches=10,
            # detect_anomaly=True,
            # profiler="simple",
        )
    else:
        trainer = Trainer(
            max_epochs=config.n_epochs,
            logger=tb_logger,
            resume_from_checkpoint="lightning_logs/version_21/checkpoints/epoch=29-step=60.ckpt",
        )
    trainer.fit(neural_rasterizer, dm)

    # ckpt_path="lightning_logs/version_21/epoch=29-step=60.ckpt"


if __name__ == "__main__":
    nice_print(HORSE)
    parser = get_parser_main_model()
    args = parser.parse_args()
    print(args)
    main(args)

# def cli_main():
#     cli = LightningCLI(
#         NeuralRasterizer, DeepVecFontDataModule,
#         seed_everything_default=42, save_config_overwrite=True, run=False
#     )
#     cli.trainer.fit(cli.model, datamodule=cli.datamodule)
#     cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
#
# if __name__ == "__main__":
#     nice_print(HORSE)
#     cli_main()
