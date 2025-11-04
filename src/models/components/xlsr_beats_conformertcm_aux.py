import torch.nn as nn
from src.models.components.conformer_tcm.model import MyConformer
from src.models.components.xlsr_aasist import SSLModel
from src.models.components.beats.BEATs import BEATsModel
import torch



class AuxiliaryBranch(nn.Module):
    """Back End Wrapper
    """
    def __init__(self, input_dim, out_dim, num_classes, 
                 dropout_rate):
        super(AuxiliaryBranch, self).__init__()

        # input feature dimension
        self.in_dim = input_dim
        # output embedding dimension
        self.out_dim = out_dim
        # number of output classes
        self.num_class = num_classes
        
        # dropout rate
        self.m_mcdp_rate = dropout_rate
        # a simple full-connected network for frame-level feature processing
        self.m_frame_level = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            
            nn.Linear(self.in_dim, self.in_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            
            nn.Linear(self.in_dim, self.out_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate)
        )

        # linear layer to produce output logits 
        self.m_utt_level = nn.Linear(self.out_dim, self.num_class)
        
        return

    def forward(self, feat):
        """ logits, emb_vec = back_end_emb(feat)

        input:
        ------
          feat: tensor, (batch, frame_num, feat_feat_dim)

        output:
        -------
          logits: tensor, (batch, num_output_class)
          emb_vec: tensor, (batch, emb_dim)
        
        """
        # through the frame-level network
        # (batch, frame_num, self.out_dim)
        feat_ = self.m_frame_level(feat)
        
        # average pooling -> (batch, self.out_dim)
        feat_utt = feat_.mean(1)
        
        # output linear 
        logits = self.m_utt_level(feat_utt)
        return logits, feat_utt


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
        
        # Auxiliary branch: Noise prediction
        aux_num_classes = args.get('aux_num_classes', 11)# 11 classes: 0-10
        self.auxiliary_branch = AuxiliaryBranch(args['emb_size'], args['emb_size'], aux_num_classes, 0.5)
        
    def feature_fusion(self, x_ssl_feat, x_beats_feat):
        if self.feature_fusion_mode == 'concat':
            x = torch.cat([x_ssl_feat, x_beats_feat], dim=1) #(bs,frame_number*2,feat_out_dim) (bs, 416, 256)
        else:
            raise ValueError(f"Invalid mode: {self.feature_fusion_mode}")
       #x = torch.cat([x_ssl_feat, x_beats_feat], dim=1) #(bs,frame_number*2,feat_out_dim) (bs, 416, 256)
        return x
    
    def forward(self, x, last_emb=False, aux_mode=True):
        x.requires_grad = True
        x_ssl_feat = self.front_end.extract_feat(x.squeeze(-1))
        x_ssl_feat = self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        if not self.is_beats_fixed:
            x_beats_feat = self.beats_model(x)
        else:
            with torch.no_grad():
                x_beats_feat = self.beats_model(x)

        x_beats_feat = self.LL_beats(x_beats_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)

        # Fusion features
        x = self.feature_fusion(x_ssl_feat, x_beats_feat)
        
        
        # Main tasks: ADD
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        
        # Main tasks: ADD
        main_task_out, attn_score =self.backend(x)
        if last_emb:
            return attn_score
        
        # Auxiliary tasks: Noise prediction
        if aux_mode:
            aux_task_out, _ = self.auxiliary_branch(x)
            return main_task_out, aux_task_out

        return main_task_out
