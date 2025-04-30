import torch.nn as nn
from src.models.components.conformer_tcm_reproduce.model import MyConformer
from src.models.components.xlsr_aasist import SSLModel

# class Model(nn.Module):
#     def __init__(self, args, ssl_pretrained_path):
#         super().__init__()
#         self.front_end = SSLModel(ssl_pretrained_path)
#         self.LL = nn.Linear(self.front_end.out_dim, args['emb_size'])
#         self.first_bn = nn.BatchNorm2d(num_features=1)
#         self.selu = nn.SELU(inplace=True)
#         self.backend=MyConformer(**args)
 
#     def forward(self, x):
#         x_ssl_feat = self.front_end.extract_feat(x.squeeze(-1))
#         x=self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
#         x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
#         x = self.first_bn(x)
#         x = self.selu(x)
#         x = x.squeeze(dim=1)
#         out, attn_score =self.backend(x)
#         return out
class Model(nn.Module):
    def __init__(self, args, ssl_pretrained_path):
        super().__init__()
        
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(ssl_pretrained_path)
        self.LL = nn.Linear(1024, args['emb_size'])
        #print('W2V + Conformer')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.conformer=MyConformer(**args)
    def forward(self, x, is_embedding=False, last_hidden_state=False):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        if is_embedding:
            return x_ssl_feat
        x=self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out, attn_score =self.conformer(x)
        
        return out
