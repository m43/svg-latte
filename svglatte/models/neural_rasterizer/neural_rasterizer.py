import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR

from deepsvg.schedulers.warmup import GradualWarmupScheduler
from svglatte.models.vgg_contextual_loss import VGGContextualLoss


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
        self.lr = self.optimizer_args.lr
        del self.optimizer_args.lr
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
        if self.cx_loss_w > 0.0:
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
            lr=self.lr, betas=(self.optimizer_args.beta1, self.optimizer_args.beta2),
            eps=self.optimizer_args.eps, weight_decay=self.optimizer_args.weight_decay
        )

        scheduler_warmup = {
            'scheduler': GradualWarmupScheduler(
                optimizer,
                multiplier=1.0,
                total_epoch=720, ),
            'interval': 'step'  # called after each training step
        }
        scheduler_lr = StepLR(optimizer, step_size=30, gamma=0.9)

        return [optimizer], [scheduler_warmup, scheduler_lr]

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
        if self.cx_loss_w > 0.0:
            vggcx_loss = self.vggcxlossfunc(dec_out, trg_img)
        else:
            vggcx_loss = {'cx_loss': latte.new_tensor([-1.0])}

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
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True)
        self.log(f'Loss/{subset}/img_l1_loss', self.l1_loss_w * l1_loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True)
        self.log(f'Loss/{subset}/img_perceptual_loss', self.cx_loss_w * vggcx_loss['cx_loss'],
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True)

        if batch_idx % 400 == 0:
            zipped = torch.concat(list(map(torch.cat, zip(dec_out[:32, None], trg_img[:32, None]))))
            zipped_image = torchvision.utils.make_grid(zipped)
            for logger in self.loggers:
                if type(logger) == WandbLogger:
                    logger.log_image(
                        f'Images/{subset}/{batch_idx}-output_img',
                        [zipped_image],
                        caption={0: f"loss:{loss} img_l1_loss:{l1_loss} img_vggcx_loss:{vggcx_loss['cx_loss']}"})
                # elif type(logger) == TensorBoardLogger:
                #     logger.experiment.add_image(
                #         f'Images/{subset}/{batch_idx}-output_img',
                #         zipped_image,
                #         self.current_epoch
                #     )

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

    def on_epoch_start(self):
        print('\n')
