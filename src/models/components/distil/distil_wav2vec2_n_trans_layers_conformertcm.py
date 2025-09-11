import torch.nn as nn
from src.models.components.conformer_tcm.model import MyConformer
from src.models.components.distil.xlsr_fe import My_XLSR_FE
import fairseq
import argparse
import torch
from fairseq.meters import AverageMeter

class My_Wav2vec2Base_FE(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs.get('num_layers', 12)
        self.order = kwargs.get('order', 'first')
        self.custom_order = kwargs.get('custom_order', None)
        ckpt_path = "/home/hungdx/wav2vec_small.pt"
        if self.num_layers < 1 or self.num_layers > 12:
            raise ValueError(
                "Number of layers must be at least 1 and at most 12.")
        #global_lists = [AverageMeter, argparse.Namespace]

        torch.serialization.add_safe_globals([argparse.Namespace])
        torch.serialization.add_safe_globals([AverageMeter])
        #torch.serialization.safe_globals(global_lists)
        #with torch.serialization.safe_globals([argparse.Namespace, AverageMeter]):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0]
        
        #self.model = self.model.to(device)
        self.out_dim = 768
        
        # self.model.eval() # Disable dropout
  
        if self.order == 'last':
            # Get the last n layers
            self.model.encoder.layers = self.model.encoder.layers[-self.num_layers:]
        elif self.order == 'first':
            # Get the first n layers
            self.model.encoder.layers = self.model.encoder.layers[:self.num_layers]
        elif self.order == 'middle':
            indices = middle_indices(12, self.num_layers)

            self.model.encoder.layers = nn.ModuleList([
                self.model.encoder.layers[i] for i in indices])
        else:
            if self.custom_order is None:
                raise ValueError(
                    "Custom order must be provided as a list of integers (0-11).")

            # Check if the custom order is valid
            if type(self.custom_order) != list:
                raise ValueError("Custom order must be a list of integers.")

            # if len(self.custom_order) != self.num_layers:
            #     raise ValueError(
            #         "Length of custom order must be less than or equal to the number of layers.")
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

class Model(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.front_end = My_Wav2vec2Base_FE(**kwargs)
        self.LL = nn.Linear(self.front_end.out_dim, args['emb_size'])
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.backend=MyConformer(**args)
    
    def forward(self, x):
        x_ssl_feat = self.front_end.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)
        x = x.unsqueeze(dim=1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out,_ = self.backend(x)
        return out