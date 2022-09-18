from deepsvg.configs.deepsvg.default_icons import *


class ModelConfig(Hierarchical):
    def __init__(self):
        super().__init__()

        self.label_condition = False
        self.use_vae = False


class Config(Config):
    def __init__(self, num_gpus=1):
        super().__init__(num_gpus=num_gpus)

        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        self.filter_category = None

        self.learning_rate = 1e-4 * num_gpus
        # self.batch_size = 3 * num_gpus
        self.batch_size = 60 * num_gpus
        # self.batch_size = 240 * num_gpus

        # self.num_epochs = 10
        self.train_ratio = 0.999
        self.val_every = 1000
        # self.pretrained_path = "logs/models/deepsvg/hierarchical_ordered/000020.pth.tar"
        # self.pretrained_path = "pretrained/hierarchical_ordered.pth.tar"

        self.data_dir = "./data/deepsvg_icons_svgtensordataset/icons_tensor/"
        self.meta_filepath = "./data/deepsvg_icons_svgtensordataset/icons_meta.csv"

    def make_schedulers(self, optimizers, epoch_size):
        optimizer, = optimizers
        return [lr_scheduler.StepLR(optimizer, step_size=5 * epoch_size, gamma=0.9)]
