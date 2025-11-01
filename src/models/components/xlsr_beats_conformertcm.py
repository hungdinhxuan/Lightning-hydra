import torch.nn as nn
from src.models.components.conformer_tcm.model import MyConformer
from src.models.components.xlsr_aasist import SSLModel
from src.models.components.beats.BEATs import BEATsModel
import torch

class Model(nn.Module):
    def __init__(self, args, ssl_pretrained_path, beats_pretrained_path):
        super().__init__()
        self.front_end = SSLModel(ssl_pretrained_path)
        self.LL = nn.Linear(self.front_end.out_dim, args['emb_size'])
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.backend=MyConformer(**args)
        self.beats_model = BEATsModel(beats_pretrained_path)
        self.LL_beats = nn.Linear(768, args['emb_size'])
        self.feature_fusion_mode = args.get('feature_fusion_mode', 'concat')
        self.is_beats_fixed = args.get('is_beats_fixed', False)
        if self.is_beats_fixed:
            print("Freezing BEATs model parameters")
            self.beats_model.eval()
            self.beats_model.freeze_model()
    
    def feature_fusion(self, x_ssl_feat, x_beats_feat):
        if self.feature_fusion_mode == 'concat':
            x = torch.cat([x_ssl_feat, x_beats_feat], dim=1) #(bs,frame_number*2,feat_out_dim) (bs, 416, 256)
        else:
            raise ValueError(f"Invalid mode: {self.feature_fusion_mode}")
       #x = torch.cat([x_ssl_feat, x_beats_feat], dim=1) #(bs,frame_number*2,feat_out_dim) (bs, 416, 256)
        return x
    
    def forward(self, x, last_emb=False):
        x.requires_grad = True
        x_ssl_feat = self.front_end.extract_feat(x.squeeze(-1))
        x_ssl_feat = self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        if not self.is_beats_fixed:
            x_beats_feat = self.beats_model(x)
        else:
            with torch.no_grad():
                x_beats_feat = self.beats_model.model(x)

        x_beats_feat = self.LL_beats(x_beats_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        #import pdb; pdb.set_trace()

        # Concat ssl feat and beats feat
        x = self.feature_fusion(x_ssl_feat, x_beats_feat)
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out, attn_score =self.backend(x)
        if last_emb:
            return attn_score
        return out
