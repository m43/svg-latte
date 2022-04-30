import argparse
from datetime import datetime

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.loggers import WandbLogger

from train import Decoder
from train import NeuralRasterizer
from svglatte.dataset import argoverse_dataset
from svglatte.dataset import deepsvg_dataset
from svglatte.dataset import deepvecfont_dataset
from svglatte.models.custom_lstms import SequenceEncoder
from svglatte.models.lstm_layernorm import LayerNormLSTMEncoder
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
    parser.add_argument('--cx_loss_w', type=float, default=0.1, help='the weight of contextual loss')
    parser.add_argument('--l1_loss_w', type=float, default=1, help='the weight of image reconstruction l1 loss')

    # encoder
    parser.add_argument('--encoder_type', type=str, choices=[
        'dvf_lstm', 'lstm', 'fc_lstm_original', 'fc_lstm', 'residual_lstm', 'lstm+mha',
    ])
    parser.add_argument('--latte_ingredients', type=str, default='hc', choices=['h', 'c', 'hc'])
    # parser.add_argument('--lstm_input_size', type=int, default=512, help='lstm encoder input_size')
    parser.add_argument('--lstm_hidden_size', type=int, default=512, help='lstm encoder hidden_size')
    parser.add_argument('--lstm_num_layers', type=int, default=4, help='svg encoder number of hidden layers')
    parser.add_argument('--lstm_dropout', type=float, default=0.0, help='lstm dropout')
    parser.add_argument('--lstm_bidirectional', action='store_true', help='')
    parser.add_argument('--no_layernorm', action='store_true', help='')
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
    parser.add_argument('--argoverse_augment_train', action='store_true',
                        help='Add augmentations to train dataset.')
    parser.add_argument('--argoverse_augment_shear_degrees', type=float, default=20.0,
                        help='Angle in degrees for random shear augmentation: (x,y)-->(x+y*tan(angle), y).'
                             ' Augmentation angle will be taken uniformly at random in [0, angle>.')
    parser.add_argument('--argoverse_augment_rotate_degrees', type=float, default=180.0,
                        help='Angle in degrees for random rotation augmentation.'
                             ' Augmentation angle will be taken uniformly at radnom in [-angle,angle>.')
    parser.add_argument('--argoverse_augment_scale_min', type=float, default=0.6,
                        help='Factor for minimum augmentation scaling.'
                             ' Scaling factor is taken uniformly at random in [augment_scale_min, augment_scale_max].')
    parser.add_argument('--argoverse_augment_scale_max', type=float, default=1.1,
                        help='Factor for maximum augmentation scaling.'
                             ' Scaling factor is taken uniformly at random in [augment_scale_min, augment_scale_max].')
    parser.add_argument('--argoverse_augment_translate', type=float, default=5.4,
                        help='Magnitude of random translations. Augmentation will make a random translation (dx, dy)'
                             ' where dx and dy are taken independently and uniformly at random'
                             ' in [-translate, translate].')
    parser.add_argument('--argoverse_numericalize', action="store_true",
                        help='Magnitude of random translations. Augmentation will make a random translation (dx, dy)'
                             ' where dx and dy are taken independently and uniformly at random'
                             ' in [-translate, translate].')
    return parser


def get_dataset(config):
    if config.dataset == "deepvecfont":
        seq_feature_dim = deepvecfont_dataset.SEQ_FEATURE_DIM
        dm = deepvecfont_dataset.DeepVecFontDataModule(
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
            augment_train=config.argoverse_augment_train,
            augment_shear_degrees=config.argoverse_augment_shear_degrees,
            augment_rotate_degrees=config.argoverse_augment_rotate_degrees,
            augment_scale_min=config.argoverse_augment_scale_min,
            augment_scale_max=config.argoverse_augment_scale_max,
            augment_translate=config.argoverse_augment_translate,
            numericalize=config.argoverse_numericalize,
        )
        seq_feature_dim = dm.train_ds.get_number_of_sequence_dimensions()
    else:
        raise Exception(f"Invalid dataset passed: {config.dataset}")
    return dm, seq_feature_dim


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
        "use_layernorm": not config.no_layernorm,
        "mha_num_layers": config.mha_num_layers,
        "mha_hidden_size": config.mha_hidden_size,
        "mha_num_heads": config.mha_num_heads,
        "mha_dropout": config.mha_dropout,
    })
    if config.encoder_type == "dvf_lstm":
        encoder_args.unpack_output = config.dvf_lstm_unpack_output
        encoder = LayerNormLSTMEncoder(**encoder_args)
    else:
        encoder = SequenceEncoder(**encoder_args)

    return encoder


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
    elif config.argoverse_rendered_images_width == 512 and config.argoverse_rendered_images_height == 512:
        decoder = Decoder(
            in_channels=encoder.output_size,
            out_channels=1,
            kernel_size_list=(3, 3, 3, 3, 3, 5, 5, 5, 5),
            stride_list=(2, 2, 2, 2, 2, 2, 2, 2, 2),
            padding_list=(1, 1, 1, 1, 1, 2, 2, 2, 2),
            output_padding_list=(1, 1, 1, 1, 1, 1, 1, 1, 1),
            ngf=4,  # 64
            # norm_layer=nn.LayerNorm,
        )
    else:
        raise Exception(f"Image size {config.argoverse_rendered_images_width}x{config.argoverse_rendered_images_height}"
                        f" not supported")
    assert config.argoverse_rendered_images_width == decoder.image_sizes[-1]
    assert config.argoverse_rendered_images_height == decoder.image_sizes[-1]

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
        config.experiment_version = f"s5_{config.dataset[:2]}" \
                                    f"_{'aug' if config.argoverse_augment_train else ''}" \
                                    f"_{config.encoder_type}" \
                                    f"{'_bi' if config.lstm_bidirectional else ''}" \
                                    f"_h={config.lstm_hidden_size}" \
                                    f"_i-{config.latte_ingredients}" \
                                    f"_l={config.lstm_num_layers}" \
                                    f"{'_noPS' if config.no_sequence_packing else ''}" \
                                    f"{'_noLN' if config.no_layernorm else ''}" \
                                    f"_e={config.n_epochs}" \
                                    f"_b={config.batch_size}" \
                                    f"_lr={config.lr}" \
                                    f"_wd={config.weight_decay}" \
                                    f"_cx={config.cx_loss_w}" \
                                    f"_l1={config.l1_loss_w}" \
                                    f"_{datetime.now().strftime('%m.%d_%H.%M.%S')}"

    wandb_logger = WandbLogger(
        project=config.experiment_name,
        version=config.experiment_version.replace("=", "-"),
        settings=wandb.Settings(start_method='thread'),
    )
    wandb_logger.watch(neural_rasterizer)
    tb_logger = TensorBoardLogger("logs", name=config.experiment_name, version=config.experiment_version, )
    csv_logger = CSVLogger("logs", name=config.experiment_name, version=config.experiment_version, )
    loggers = [wandb_logger, tb_logger, csv_logger]
    # loggers = []

    if torch.cuda.is_available() and config.gpus != 0:
        # strategy = 'dp' if config.cx_loss_w > 0.0 else DDPStrategy(find_unused_parameters=False),

        trainer = Trainer(
            max_epochs=config.n_epochs,
            default_root_dir="logs",
            logger=loggers,
            callbacks=[
                EarlyStopping(monitor="Loss/val/loss", mode="min", patience=72, check_on_train_epoch_end=False),
                ModelCheckpoint(monitor="Loss/val/loss", save_last=True),
            ],
            gpus=config.gpus,
            accelerator="gpu",
            # strategy=strategy,
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
