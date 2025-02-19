'''
This file contains the unofficial implementation of the neural network model proposed in the paper:
"Exploring Self-supervised Embeddings and Synthetic Data Augmentation for Robust Audio Deepfake Detection"
https://digibug.ugr.es/bitstream/handle/10481/97026/martindonas24_interspeech.pdf?sequence=1&isAllowed=y
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.pooling import ASP
from src.models.components.xlsr_aasist_layerwise import SSLModel

class FrameProcessor(nn.Module):
    def __init__(self, input_dim=768, output_dim=256, use_nn=False, dropout=0.1):
        super().__init__()
        if use_nn:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(output_dim, output_dim)
            )
        else:
            self.proj = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.proj(x)

class Model(nn.Module):
    def __init__(self, ssl_pretrained_path, n_layers, frame_processor_type='proj'):
        super().__init__()
        self.ssl_model = SSLModel(ssl_pretrained_path, n_layers)
        self.layer_norm = nn.BatchNorm2d(num_features=n_layers)
        self.weight_hidd = nn.Parameter(torch.ones(n_layers))
        
        # Frame processor
        self.frame_processor = FrameProcessor(
            input_dim=self.ssl_model.out_dim,  # Assuming SSL model output dim
            output_dim=256,
            use_nn=(frame_processor_type == 'nn')
        )
        
        # ASP layer
        self.asp = ASP(out_dim=256, input_dim=256)
        
        # Scoring layer
        self.score_projection = nn.Linear(512, 128)  # 512 = 256 * 2 (mean and variance)
        self.w = nn.Parameter(torch.randn(128))
        
    def forward(self, x, attention_mask=None):
        # SSL feature extraction
        x_ssl_feat = self.ssl_model.extract_feat(x)
        x_ssl_feat = self.layer_norm(x_ssl_feat)
        norm_weights = F.softmax(self.weight_hidd, dim=-1)
        weighted_feature = (x_ssl_feat * norm_weights.view(-1, 1, 1)).sum(dim=1)
        
        # Frame processing
        processed_frames = self.frame_processor(weighted_feature)
        
        # ASP pooling
        if attention_mask is None:
            attention_mask = torch.zeros(processed_frames.size(0), processed_frames.size(1)).to(processed_frames.device)
        pooled_features = self.asp(processed_frames, attention_mask)
        
        # Scoring
        projected_features = self.score_projection(pooled_features)
        normalized_features = F.normalize(projected_features, p=2, dim=-1)
        normalized_w = F.normalize(self.w, p=2, dim=-1)
        score = torch.sum(normalized_features * normalized_w, dim=-1)
        
        return score