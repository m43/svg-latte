from torch.optim import lr_scheduler

from deepsvg.configs.deepsvg.default_icons import Config, Hierarchical


class ModelConfig(Hierarchical):
    def __init__(self):
        super().__init__()

        self.label_condition = False
        self.use_vae = False


class Config(Config):
    def __init__(self, num_gpus=2):
        super().__init__(num_gpus=num_gpus)

        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        self.filter_category = None

        self.learning_rate = 1e-4 * num_gpus
        self.batch_size = 80 * num_gpus

        # self.num_epochs = 100
        self.num_epochs = 150
        self.log_every = 20
        self.val_every = 1000
        self.ckpt_every = 1000
        # self.pretrained_path = "logs2/models/dataset/deepsvg_hierarchical_ordered_argoverse_6/006863.pth.tar"

        # Argoverse specific
        self.max_num_groups = 100
        self.max_seq_len = 10
        self.max_total_len = 261
        self.nb_augmentations = 1

        self.model_cfg.max_num_groups = self.max_num_groups
        self.model_cfg.max_seq_len = self.max_seq_len
        self.model_cfg.max_total_len = self.max_total_len
        self.model_cfg.num_groups_proposal = self.max_num_groups

        self.dataloader_module = "svglatte.train_deepsvg"

        self.train_meta_filepath = "./data/argoverse/svgdataset/train/svg_meta.csv"
        self.train_data_dir = "./data/argoverse/svgdataset/train/svgs_simplified/"

        self.val_meta_filepath = "./data/argoverse/svgdataset/val/svg_meta.csv"
        self.val_data_dir = "./data/argoverse/svgdataset/val/svgs_simplified/"

        self.test_meta_filepath = "./data/argoverse/svgdataset/test/svg_meta.csv"
        self.test_data_dir = "./data/argoverse/svgdataset/test/svgs_simplified/"

    def make_schedulers(self, optimizers, epoch_size):
        optimizer, = optimizers
        return [lr_scheduler.StepLR(optimizer, step_size=5 * epoch_size, gamma=0.9)]
