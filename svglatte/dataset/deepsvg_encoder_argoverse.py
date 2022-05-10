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

        # self.pretrained_path = "/work/vita/frano/checkpoints/" \
        #                        "deepsvg_hierarchical_ordered_argoverse_2_FIXED_LR/best.pth.tar"

        # Argoverse specific
        self.max_num_groups = 15
        self.max_seq_len = 30
        self.max_total_len = self.max_num_groups * self.max_seq_len

        self.model_cfg.max_num_groups = self.max_num_groups
        self.model_cfg.max_seq_len = self.max_seq_len
        self.model_cfg.max_total_len = self.max_total_len
        self.model_cfg.num_groups_proposal = self.max_num_groups
