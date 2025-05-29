import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from src.models.components.conformer_tcm.model import MyConformer
from src.models.components.xlsr_aasist_layerwise import SSLModel


class Model(nn.Module):
    def __init__(self, args, ssl_pretrained_path, n_layers, extractor_type='layerwise'):
        super().__init__()
        self.front_end = SSLModel(ssl_pretrained_path, n_layers, extractor_type)
        self.n_layers = self.front_end.n_layers
        self.extractor_type = extractor_type
        if extractor_type == 'layerwise':
            self.layer_norm = nn.BatchNorm2d(num_features=self.n_layers)
            self.weight_hidd = nn.Parameter(torch.ones(self.n_layers))
        self.LL = nn.Linear(self.front_end.out_dim, args['emb_size'])
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.backend=MyConformer(**args)
 
    def forward(self, x, last_emb=False):
        x.requires_grad = True
        x_ssl_feat = self.front_end.extract_feat(x.squeeze(-1))
        
        if self.extractor_type == 'layerwise':    
            x_ssl_feat = self.layer_norm(x_ssl_feat)
            norm_weights = F.softmax(self.weight_hidd, dim=-1)
            weighted_feature = (
                x_ssl_feat * norm_weights.view(-1, 1, 1)).sum(dim=1)
            
            x=self.LL(weighted_feature) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        else:
            x=self.LL(x_ssl_feat)
            
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out, attn_score =self.backend(x)
        if last_emb:
            return attn_score
        return out
