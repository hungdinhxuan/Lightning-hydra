import torch.nn as nn
import os
import fairseq

class My_XLSR_FE(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs.get('num_layers', 24)
        self.order = kwargs.get('order', 'first')
        self.custom_order = kwargs.get('custom_order', None)
        ckpt_path = os.getenv("XLSR_PRETRAINED_PATH", "/nvme1/hungdx/Lightning-hydra/xlsr2_300m.pt")
        if self.num_layers < 1 or self.num_layers > 24:
            raise ValueError(
                "Number of layers must be at least 1 and at most 24.")
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0]
        #self.model = self.model
        self.out_dim = 1024

        if self.order == 'last':
            # Get the last n layers
            self.model.encoder.layers = self.model.encoder.layers[-self.num_layers:]
        elif self.order == 'first':
            # Get the first n layers
            self.model.encoder.layers = self.model.encoder.layers[:self.num_layers]
        elif self.order == 'middle':
            indices = middle_indices(24, self.num_layers)

            self.model.encoder.layers = nn.ModuleList([
                self.model.encoder.layers[i] for i in indices])
        else:
            if self.custom_order is None:
                raise ValueError(
                    "Custom order must be provided as a list of integers (0-23).")

            self.model.encoder.layers = nn.ModuleList([
                self.model.encoder.layers[i] for i in self.custom_order])

    def forward(self, x):
        return self.extract_feat(x)

    def extract_feat(self, x):
        input_tmp = x[:, :, 0] if x.ndim == 3 else x
        emb = self.model(input_tmp, mask=False, features_only=True)[
            'x']
        return emb

    def extract_layer_results(self, x):
        input_tmp = x[:, :, 0] if x.ndim == 3 else x
        layer_results = self.model(input_tmp, mask=False, features_only=True)[
            'layer_results']
        return layer_results
