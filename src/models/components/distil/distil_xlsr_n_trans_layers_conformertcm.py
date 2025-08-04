import torch.nn as nn
from src.models.components.conformer_tcm.model import MyConformer
from src.models.components.distil.xlsr_fe import My_XLSR_FE

class Model(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.front_end = My_XLSR_FE(**kwargs)
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