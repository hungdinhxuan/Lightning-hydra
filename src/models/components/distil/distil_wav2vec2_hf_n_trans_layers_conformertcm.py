import torch.nn as nn
from src.models.components.conformer_tcm.model import MyConformer
from src.models.components.distil.xlsr_fe import My_XLSR_FE
import fairseq
import argparse
import torch
from fairseq.meters import AverageMeter
from transformers import Wav2Vec2Model
def middle_indices(array_length, number_of_middle_elements):
    # Calculate the start index
    start_index = (array_length - number_of_middle_elements) // 2
    # Calculate the end index
    end_index = start_index + number_of_middle_elements
    # Create a list of the middle indices
    middle_indices = list(range(start_index, end_index))
    return middle_indices
class My_Wav2vec2Base_FE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.num_layers = kwargs.get('num_layers', 12)
        self.order = kwargs.get('order', 'first')
        self.custom_order = kwargs.get('custom_order', None)
        self.out_dim = 768
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
                    "Custom order must be provided as a list of integers (0-23).")

            # Check if the custom order is valid
            if type(self.custom_order) != list:
                raise ValueError("Custom order must be a list of integers.")

            # if len(self.custom_order) != self.num_layers:
            #     raise ValueError(
            #         "Length of custom order must be less than or equal to the number of layers.")
            self.model.encoder.layers = nn.ModuleList([
                self.model.encoder.layers[i] for i in self.custom_order])


    def forward(self, x):
        x = self.model(x).last_hidden_state
        return x

class Model(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.front_end = My_Wav2vec2Base_FE(**kwargs)
        self.LL = nn.Linear(self.front_end.out_dim, args['emb_size'])
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.backend=MyConformer(**args)
    
    def forward(self, x):
        x_ssl_feat = self.front_end(x)
        x = self.LL(x_ssl_feat)
        x = x.unsqueeze(dim=1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out,_ = self.backend(x)
        return out