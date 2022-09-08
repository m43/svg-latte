import argparse
import importlib
import warnings
from datetime import datetime

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.loggers import WandbLogger

from svglatte.dataset import argoverse_dataset
from svglatte.dataset import deepsvg_dataset
from svglatte.dataset import deepvecfont_dataset
from svglatte.models.custom_lstms import SequenceEncoder
from svglatte.models.deepsvg_encoder import DeepSVGEncoder
from svglatte.models.lstm_layernorm import LayerNormLSTMEncoder
from svglatte.models.neural_rasterizer.cnn_decoder import Decoder
from svglatte.models.neural_rasterizer.neural_rasterizer import NeuralRasterizer
from svglatte.models.sequence_average_encoder import SequenceAverageEncoder
from svglatte.utils.util import AttrDict, nice_print, HORSE


def get_parser_main_model():
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--experiment_name', type=str, default='svglatte')
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Checkpoint used to restore training state")
    parser.add_argument('--experiment_version', type=str, default=None)
    parser.add_argument('--do_not_add_timestamp_to_experiment_version', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--early_stopping_patience', type=int, default=72)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--training_strategy', type=str, default="dp")
    parser.add_argument('--seq_feature_dim', type=int, default=8)

    # optimizer
    parser.add_argument('--encoder_lr', type=float, default=0.00042, help='encoder learning rate')
    parser.add_argument('--encoder_weight_decay', type=float, default=0.0, help='encoder weight decay')
    parser.add_argument('--decoder_lr', type=float, default=0.00042, help='decoder learning rate')
    parser.add_argument('--decoder_weight_decay', type=float, default=0.0, help='decoder weight decay')
    parser.add_argument('--auto_lr_find', action='store_true', help='Use the auto lr finder from Pytorch Lightning')
    parser.add_argument('--warmup_steps', type=int, default=720, help='number of warmup steps')
    parser.add_argument('--scheduler_decay_epochs', type=int, default=30,
                        help='Number of steps to decay learning rate, periodically')
    parser.add_argument('--scheduler_decay_gamma', type=float, default=0.9, help='Factor for learning rate decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam optimizer')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--gradient_clip_val', type=float, default=None, help='gradient clipping value')

    # loss weight
    parser.add_argument('--cx_loss_w', type=float, default=0.1, help='the weight of contextual loss')
    parser.add_argument('--l1_loss_w', type=float, default=1, help='the weight of image reconstruction l1 loss')

    # encoder
    parser.add_argument('--encoder_type', type=str, choices=[
        'dvf_lstm', 'lstm', 'fc_lstm_original', 'fc_lstm', 'residual_lstm', 'lstm+mha', 'deepsvg', 'averager_baseline'
    ])
    parser.add_argument('--latte_ingredients', type=str, default='hc', choices=['h', 'c', 'hc'],
                        help="For LSTM based encoders, should the latent representation contain"
                             "only the hidden state (h), only the cell state (c), or both (hc)")
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
    parser.add_argument('--deepsvg_encoder_config_module', type=str,
                        default='svglatte.dataset.deepsvg_encoder_argoverse')
    parser.add_argument('--deepsvg_encoder_hidden_size', type=int, default=256,
                        help='Latent representation dimensionality of DeepSVG\'s encoder.'
                             'The value will be written into the `dim_z` parameter of the model.')

    # decoder
    parser.add_argument('--decoder_n_filters_in_last_conv_layer', type=int, default=16, help='')
    parser.add_argument('--decoder_norm_layer_name', type=str, default="layernorm",
                        help='Which layer normalization to use between the deconvolution layers.')

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
    parser.add_argument('--argoverse_sequences_format', type=str,
                        default="onehot_grouped_commands_concatenated_with_grouped_arguments",
                        choices=["onehot_grouped_commands_concatenated_with_grouped_arguments", "svgtensor_data"],
                        help="What is the format of the tensors in the cached sequences.")
    parser.add_argument('--argoverse_train_sequences_path', type=str,
                        default='data/argoverse/train.sequences.torchsave')
    parser.add_argument('--argoverse_val_sequences_path', type=str, default='data/argoverse/val.sequences.torchsave')
    parser.add_argument('--argoverse_test_sequences_path', type=str, default='data/argoverse/test.sequences.torchsave')
    parser.add_argument('--argoverse_rendered_images_width', type=int, default=64, help='Height of rendered images')
    parser.add_argument('--argoverse_rendered_images_height', type=int, default=64, help='Width of rendered images')
    parser.add_argument('--argoverse_fast_run', action='store_true',
                        help='To gave a faster run, we use the smallest dataset subset for all datasets')
    parser.add_argument('--argoverse_keep_redundant_features', action='store_true',
                        help='Keep the redundant features in svg tensor sequences. These features are redundant '
                             'because (1.) DeepSVG svg tensors have a few unused features, we keep them internally to'
                             'be able to use the DeepSVG library, (2.) we leave only lines when preprocessing'
                             'Argoverse (curves are simplfied into lines).')
    parser.add_argument('--argoverse_train_workers', type=int, default=4, help='')
    parser.add_argument('--argoverse_val_workers', type=int, default=4, help='')
    parser.add_argument('--argoverse_test_workers', type=int, default=0, help='')
    parser.add_argument('--argoverse_zoom_preprocess_factor', type=float, default=1.0,
                        help='Factor for scaling during preprocessing. Useful when all SVGs must be zoomed out.')
    parser.add_argument('--argoverse_viewbox', type=int, default=24,
                        help='The size of the viewbox to be used when rendering the SVG using cairosvg.'
                             'The larger the viewbox, the thinner the rendered lines become.')
    parser.add_argument('--argoverse_augment_train', action='store_true', help='Add augmentations to train dataset.')
    parser.add_argument('--argoverse_augment_val', action='store_true', help='Add augmentations to val dataset.')
    parser.add_argument('--argoverse_augment_test', action='store_true', help='Add augmentations to test dataset.')
    parser.add_argument('--argoverse_augment_shear_degrees', type=float, default=0.0,
                        help='Angle in degrees for random shear augmentation: (x,y)-->(x+y*tan(angle), y).'
                             ' Augmentation angle will be taken uniformly at random in [0, angle>.')
    parser.add_argument('--argoverse_augment_rotate_degrees', type=float, default=180.0,
                        help='Angle in degrees for random rotation augmentation.'
                             ' Augmentation angle will be taken uniformly at radnom in [-angle,angle>.')
    parser.add_argument('--argoverse_augment_scale_min', type=float, default=1.0,
                        help='Factor for minimum augmentation scaling.'
                             ' Scaling factor is taken uniformly at random in [augment_scale_min, augment_scale_max].')
    parser.add_argument('--argoverse_augment_scale_max', type=float, default=1.0,
                        help='Factor for maximum augmentation scaling.'
                             ' Scaling factor is taken uniformly at random in [augment_scale_min, augment_scale_max].')
    parser.add_argument('--argoverse_augment_translate', type=float, default=0.0,
                        help='Magnitude of random translations. Augmentation will make a random translation (dx, dy)'
                             ' where dx and dy are taken independently and uniformly at random'
                             ' in [-translate, translate].')
    parser.add_argument('--argoverse_numericalize', action="store_true",
                        help='Magnitude of random translations. Augmentation will make a random translation (dx, dy)'
                             ' where dx and dy are taken independently and uniformly at random'
                             ' in [-translate, translate].')
    parser.add_argument('--argoverse_canonicalize_svg', action="store_true",
                        help='Canonicalize the SVG during preprocessing.')
    return parser


def get_dataset(config):
    deepsvg_encoder_config = None
    deepsvg_encoder_config_for_argoverse = {}
    if config.encoder_type == "deepsvg":
        deepsvg_encoder_config = importlib.import_module(config.deepsvg_encoder_config_module).Config()
        deepsvg_encoder_config.model_cfg.dim_z = config.deepsvg_encoder_hidden_size
        deepsvg_encoder_config_for_argoverse.update({
            'return_deepsvg_model_input': True,
            'deepsvg_model_args': deepsvg_encoder_config.model_args,
            'deepsvg_max_num_groups': deepsvg_encoder_config.max_num_groups,
            'deepsvg_max_seq_len': deepsvg_encoder_config.max_seq_len,
            'deepsvg_max_total_len': deepsvg_encoder_config.max_total_len,
            'deepsvg_pad_val': -1,
        })
    else:
        deepsvg_encoder_config_for_argoverse.update({
            'return_deepsvg_model_input': False,
        })

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
            sequences_format=config.argoverse_sequences_format,
            train_sequences_path=config.argoverse_train_sequences_path,
            val_sequences_path=config.argoverse_val_sequences_path,
            test_sequences_path=config.argoverse_test_sequences_path,
            rendered_images_width=config.argoverse_rendered_images_width,
            rendered_images_height=config.argoverse_rendered_images_height,
            batch_size=config.batch_size,
            remove_redundant_features=not config.argoverse_keep_redundant_features,
            fast_run=config.argoverse_fast_run,
            train_workers=config.argoverse_train_workers,
            val_workers=config.argoverse_val_workers,
            test_workers=config.argoverse_test_workers,
            zoom_preprocess_factor=config.argoverse_zoom_preprocess_factor,
            viewbox=config.argoverse_viewbox,
            augment_train=config.argoverse_augment_train,
            augment_val=config.argoverse_augment_val,
            augment_test=config.argoverse_augment_test,
            augment_shear_degrees=config.argoverse_augment_shear_degrees,
            augment_rotate_degrees=config.argoverse_augment_rotate_degrees,
            augment_scale_min=config.argoverse_augment_scale_min,
            augment_scale_max=config.argoverse_augment_scale_max,
            augment_translate=config.argoverse_augment_translate,
            numericalize=config.argoverse_numericalize,
            canonicalize_svg=config.argoverse_canonicalize_svg,
            **deepsvg_encoder_config_for_argoverse
        )
        # TODO:
        #  The DataModule now uses setup which we do not want to call only to figure out the seq_feature_dim value.
        #  Quick fix is to pass the value of seq_feature_dim through the command line, like `--seq_feature_dim 8`.
        # dm.setup()
        # seq_feature_dim = dm.train_ds.get_number_of_sequence_dimensions()
        # print("seq_feature_dim", seq_feature_dim)
        seq_feature_dim = config.seq_feature_dim
    else:
        raise Exception(f"Invalid dataset passed: {config.dataset}")
    return dm, seq_feature_dim, deepsvg_encoder_config


def get_encoder(config, deepsvg_cfg=None):
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
    elif config.encoder_type == "deepsvg":
        encoder = DeepSVGEncoder(deepsvg_cfg)
    elif config.encoder_type == "averager_baseline":
        encoder = SequenceAverageEncoder(encoder_args.lstm_input_size)
    else:
        encoder = SequenceEncoder(**encoder_args)

    return encoder


def get_decoder(config, encoder_output_size):
    if config.argoverse_rendered_images_width == 64 and config.argoverse_rendered_images_height == 64:
        decoder = Decoder(
            in_channels=encoder_output_size,
            out_channels=1,
            norm_layer_name=config.decoder_norm_layer_name,
            n_filters_in_last_conv_layer=config.decoder_n_filters_in_last_conv_layer,
        )
    elif config.argoverse_rendered_images_width == 128 and config.argoverse_rendered_images_height == 128:
        decoder = Decoder(
            in_channels=encoder_output_size,
            out_channels=1,
            kernel_size_list=(3, 3, 3, 5, 5, 5, 5),
            stride_list=(2, 2, 2, 2, 2, 2, 2),
            padding_list=(1, 1, 1, 2, 2, 2, 2),
            output_padding_list=(1, 1, 1, 1, 1, 1, 1),
            norm_layer_name=config.decoder_norm_layer_name,
            n_filters_in_last_conv_layer=config.decoder_n_filters_in_last_conv_layer,
        )
    elif config.argoverse_rendered_images_width == 200 and config.argoverse_rendered_images_height == 200:
        decoder = Decoder(
            in_channels=encoder_output_size,
            out_channels=1,
            kernel_size_list=(3, 3, 3, 3, 3, 3, 5, 5),
            stride_list=(2, 2, 2, 2, 2, 2, 2, 2),
            padding_list=(1, 1, 1, 2, 2, 2, 2, 2),
            output_padding_list=(1, 1, 1, 1, 1, 1, 1, 1),
            norm_layer_name=config.decoder_norm_layer_name,
            n_filters_in_last_conv_layer=config.decoder_n_filters_in_last_conv_layer,  # 64
        )
    elif config.argoverse_rendered_images_width == 256 and config.argoverse_rendered_images_height == 256:
        decoder = Decoder(
            in_channels=encoder_output_size,
            out_channels=1,
            kernel_size_list=(3, 3, 3, 5, 5, 5, 5, 5),
            stride_list=(2, 2, 2, 2, 2, 2, 2, 2),
            padding_list=(1, 1, 1, 2, 2, 2, 2, 2),
            output_padding_list=(1, 1, 1, 1, 1, 1, 1, 1),
            norm_layer_name=config.decoder_norm_layer_name,
            n_filters_in_last_conv_layer=config.decoder_n_filters_in_last_conv_layer,
        )
    elif config.argoverse_rendered_images_width == 512 and config.argoverse_rendered_images_height == 512:
        decoder = Decoder(
            in_channels=encoder_output_size,
            out_channels=1,
            kernel_size_list=(3, 3, 3, 3, 3, 5, 5, 5, 5),
            stride_list=(2, 2, 2, 2, 2, 2, 2, 2, 2),
            padding_list=(1, 1, 1, 1, 1, 2, 2, 2, 2),
            output_padding_list=(1, 1, 1, 1, 1, 1, 1, 1, 1),
            norm_layer_name=config.decoder_norm_layer_name,
            n_filters_in_last_conv_layer=config.decoder_n_filters_in_last_conv_layer,  # 64
        )
    else:
        raise Exception(
            f"Image size {config.argoverse_rendered_images_width}x{config.argoverse_rendered_images_height}"
            f" not supported")
    return decoder


def get_neural_rasterizer(config, deepsvg_encoder_config=None):
    encoder = get_encoder(config, deepsvg_encoder_config)
    decoder = get_decoder(config, encoder.output_size)
    adam_optimizer_args = AttrDict()
    adam_optimizer_args.update({
        # "lr": config.lr,
        "encoder_lr": config.encoder_lr,
        "encoder_weight_decay": config.encoder_weight_decay,
        "decoder_lr": config.decoder_lr,
        "decoder_weight_decay": config.decoder_weight_decay,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "eps": config.eps,
        # "weight_decay": config.weight_decay,
        "warmup_steps": config.warmup_steps,
        "scheduler_decay_epochs": config.scheduler_decay_epochs,
        "scheduler_decay_gamma": config.scheduler_decay_gamma,
    })
    neural_rasterizer = NeuralRasterizer(
        encoder=encoder,
        decoder=decoder,
        optimizer_args=adam_optimizer_args,
        l1_loss_w=config.l1_loss_w,
        cx_loss_w=config.cx_loss_w,
        dataset_name=config.dataset,
        train_mean=None,
        train_std=None,
        standardize_input_sequences=config.standardize_input_sequences,
    )
    return neural_rasterizer


def main(config):
    pl.seed_everything(config.seed, workers=True)

    dm, seq_feature_dim, deepsvg_encoder_config = get_dataset(config)
    config.lstm_input_size = seq_feature_dim

    neural_rasterizer = get_neural_rasterizer(config, deepsvg_encoder_config)
    print(neural_rasterizer)

    assert config.argoverse_rendered_images_width == neural_rasterizer.decoder.image_sizes[-1]
    assert config.argoverse_rendered_images_height == neural_rasterizer.decoder.image_sizes[-1]

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
                                    f"_l1={config.l1_loss_w}"
    if not config.do_not_add_timestamp_to_experiment_version:
        config.experiment_version += f"_{datetime.now().strftime('%m.%d_%H.%M.%S')}"

    wandb_logger = WandbLogger(
        project=config.experiment_name,
        version=config.experiment_version.replace("=", "-"),
        # settings=wandb.Settings(start_method='thread'),
        settings=wandb.Settings(start_method='fork'),
        log_model=True,
    )
    wandb_logger.watch(neural_rasterizer)
    tb_logger = TensorBoardLogger("logs", name=config.experiment_name, version=config.experiment_version, )
    csv_logger = CSVLogger("logs", name=config.experiment_name, version=config.experiment_version, )
    loggers = [wandb_logger, tb_logger, csv_logger]
    # loggers = []

    early_stopping_callback = EarlyStopping(
        monitor="Loss/val/loss", mode="min",
        patience=max(2, config.early_stopping_patience // config.check_val_every_n_epoch),
        check_on_train_epoch_end=False
    )
    model_checkpoint_callback = ModelCheckpoint(monitor="Loss/val/loss", save_top_k=1, save_last=True, verbose=True, )
    learning_rate_monitor_callback = LearningRateMonitor(logging_interval='step')
    callbacks = [
        early_stopping_callback,
        model_checkpoint_callback,
        learning_rate_monitor_callback]

    print("Configuration:")
    print(config)
    if torch.cuda.is_available() and config.gpus != 0:
        trainer = Trainer(
            max_epochs=config.n_epochs,
            check_val_every_n_epoch=config.check_val_every_n_epoch,
            default_root_dir="logs",
            logger=loggers,
            callbacks=callbacks,
            gradient_clip_val=config.gradient_clip_val,
            auto_lr_find=config.auto_lr_find,
            precision=config.precision,
            gpus=config.gpus,
            accelerator="gpu",
            strategy=config.training_strategy,

            # ~~~ Uncomment for very very fast debugging ~~~ #
            # limit_train_batches=5,
            # limit_val_batches=5,
        )
    else:
        print("\n\n")
        print("*****************")
        print("*** Using CPU ***")
        print("*****************")
        print("\n\n")
        trainer = Trainer(
            max_epochs=config.n_epochs,
            default_root_dir="logs",
            logger=[wandb_logger, tb_logger, csv_logger],
            callbacks=callbacks,
        )
    if config.auto_lr_find:
        warnings.warn("Pytorch Lightning 1.6.1. does not support multiple optimizers (or multiple"
                      "parameter group learning rates) for auto learning rate finding.")
        print(f"Calling trainer.tune() to automatically find the learning rate")
        print(f"config.auto_lr_find={config.auto_lr_find}")
        trainer.tune(neural_rasterizer, dm)
        print("Finished tuning")
    trainer.fit(neural_rasterizer, dm, ckpt_path=config.checkpoint_path)
    print(f"best_model_path={model_checkpoint_callback.best_model_path}")
    trainer.test(neural_rasterizer, dm, ckpt_path='best')


if __name__ == "__main__":
    nice_print(HORSE)
    parser = get_parser_main_model()
    args = parser.parse_args()
    print(args)
    main(args)
