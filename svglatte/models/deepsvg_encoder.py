from torch import nn

from deepsvg import utils
from deepsvg.model.model import SVGTransformer


class DeepSVGEncoder(nn.Module):
    def __init__(self, cfg, verbose=True):
        super().__init__()
        self.verbose = verbose
        if self.verbose:
            print("Loading DeepSVG")

        self.cfg = cfg
        if self.verbose:
            print("Parameters")
            self.cfg.print_params()
            print("Model Configuration:")
            for key in dir(self.cfg.model_cfg):
                if not key.startswith("__") and not callable(getattr(self.cfg.model_cfg, key)):
                    print(f"  {key} = {getattr(self.cfg.model_cfg, key)}")

        self.transformer = SVGTransformer(self.cfg.model_cfg)  # self.cfg.make_model()
        self.output_size = self.cfg.model_cfg.dim_z  # TODO

        if self.cfg.pretrained_path is not None:
            print(f"Loading pretrained model {self.cfg.pretrained_path}")
            utils.load_model(self.cfg.pretrained_path, self.transformer, "cpu")

        print(f"#Parameters: {utils.count_parameters(self.transformer)}")

    def forward(self, model_inputs, sequence_lengths=None):
        model_inputs = [model_inputs[model_arg] for model_arg in self.cfg.model_args]
        latte = self.transformer(*model_inputs, encode_mode=True)
        latte = latte.squeeze()
        assert latte.shape[-1] == self.output_size
        return latte
