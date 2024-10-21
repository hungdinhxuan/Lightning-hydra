import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones
import fairseq

from src.models.components.loss_metrics import supcon_loss
from src.models.components.conformer_tcm.model import MyConformer


class SSLModel(nn.Module):
    def __init__(self, cp_path):
        super(SSLModel, self).__init__()
        
        #cp_path = '/data/hungdx/asvspoof5/model/pretrained/xlsr2_300m.pt'   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        
        self.out_dim = 1024
        return

    def extract_feat(self, input_data, is_train=True):
        


        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
            
        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        # print(emb.shape)
        return emb
    
    def forward(self, x):
        # x is a tensor of [batch, length]
        return self.extract_feat(x)


class Model(nn.Module):
    def __init__(self, args, cp_path, is_train = True):
        super().__init__()
        
        self.is_train = is_train
        self.contra_mode = args['contra_mode']
        self.loss_type = args['loss_type']
        
        self.loss_CE = nn.CrossEntropyLoss(weight = torch.FloatTensor([float(1-args['ce_loss_weight']), float(args['ce_loss_weight'])]))

        ####
        # create network wav2vec 2.0
        ####
        self.front_end = SSLModel(cp_path)
        self.LL = nn.Linear(self.front_end.out_dim, args['conformer']['emb_size'])
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.selu = nn.SELU(inplace=True)
        
        self.backend=MyConformer(**args['conformer'])
        self.loss_CE = nn.CrossEntropyLoss()
        
        self.sim_metric_seq = lambda mat1, mat2: torch.bmm(
            mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)

    def _forward(self, x):
        x.requires_grad = True
        #-----------------RawNet3-----------------#
        x = self.front_end(x) #(bs,frame_number,frontend_out_dim)
        x = self.LL(x) #(bs,frame_number,feat_out_dim)
        
        feats = x
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)

        # output [batch, 2]
        # emb [batch, emb_size]
        output, emb = self.backend(x)
        output = F.log_softmax(output, dim=1)
        if (self.is_train):
            return output, feats, emb
        return output
    
    def forward(self, x_big):
        # make labels to be a tensor of [bz]
        # labels = labels.squeeze(0)
        if (x_big.dim() == 3):
            x_big = x_big.transpose(0,1)
            batch, length, sample_per_batch = x_big.shape
            # x_big is a tensor of [length, batch, sample per batch]
            # transform to [length, batch*sample per batch] by concat last dim
            x_big = x_big.transpose(1,2)
            x_big = x_big.reshape(batch * sample_per_batch, length)
        if (self.is_train):
            # x_big is a tensor of [1, length, bz]
            # convert to [bz, length]
            # x_big = x_big.squeeze(0).transpose(0,1)
            output, feats, emb = self._forward(x_big)
            # calculate the loss
            return output, feats, emb
        else:
            # in inference mode, we don't need the emb
            # the x_big now is a tensor of [bz, length]
            return self._forward(x_big)
        
    
    def loss(self, output, feats, emb, labels, config, info=None):
        real_bzs = output.shape[0]
        # print("real_bzs", real_bzs)
        # print("labels", labels)
        L_CE = 1/real_bzs *self.loss_CE(output, labels)
        
        # reshape the feats to match the supcon loss format
        feats = feats.unsqueeze(1)
        # print("feats.shape", feats.shape)
        L_CF1 = 1/real_bzs *supcon_loss(feats, labels=labels, contra_mode=self.contra_mode, sim_metric=self.sim_metric_seq)
        # reshape the emb to match the supcon loss format
        emb = emb.unsqueeze(1)
        emb = emb.unsqueeze(-1)
        # print("emb.shape", emb.shape)
        L_CF2 = 1/real_bzs *supcon_loss(emb, labels=labels, contra_mode=self.contra_mode, sim_metric=self.sim_metric_seq)
        if self.loss_type == 1:
            return {'L_CE':L_CE, 'L_CF1':L_CF1, 'L_CF2':L_CF2}
        elif self.loss_type == 2:
            return {'L_CE':L_CE, 'L_CF1':L_CF1}
        elif self.loss_type == 3:
            return {'L_CE':L_CE, 'L_CF2':L_CF2}
        # ablation study
        elif self.loss_type == 4:
            return {'L_CE':L_CE}
        elif self.loss_type == 5:
            return {'L_CF1':L_CF1, 'L_CF2':L_CF2}
