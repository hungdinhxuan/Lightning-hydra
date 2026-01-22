from .WavLM import WavLM, WavLMConfig
import torch.nn as nn
import torch
class SSLModelWavlm(nn.Module):
    def __init__(self, cp_path, n_layers=None):
        super(SSLModelWavlm, self).__init__()
        

        checkpoint = torch.load(cp_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        self.n_layers = n_layers if n_layers is not None else cfg.encoder_layers
        self.model = WavLM(cfg)
        print("Default config: ", cfg)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        print("Loaded model config: ", self.model.cfg.__dict__)
        print("Loaded model successfully")
        self.model.encoder.layers = self.model.encoder.layers[:self.n_layers]
        self.out_dim = cfg.encoder_embed_dim
        print("Output dimension: ", self.out_dim)
        return
    
    def forward(self, x):
        return self.extract_feat(x)

    def extract_feat(self, input_data):
        input_data = input_data.squeeze(1)
        x, layers = self.model.extract_features(input_data, mask=False, ret_layer_results=True)[0]

        return torch.stack(layers[:self.n_layers], dim=1).permute(2,1,0,3).contiguous()
    
    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    from torchinfo import summary
    model = SSLModelWavlm('/home/hungdx/code/Lightning-hydra/pretrained/WavLM-Base+.pt', None)
    x = torch.randn(1,64000)
    
    feats = model(x)
    
    # Projection layer
    proj_layer = nn.Linear(model.out_dim, 128)
    feats = proj_layer(feats)
    
    pooling_feat = torch.mean(feats, dim=1)
    
    # Another feat 
    x2 = torch.randn(48, 128)
    x2 = x2.unsqueeze(0)
    print(x2.shape)
    # concat pooling_feat and x2
    concat_feat = torch.cat([pooling_feat, x2], dim=1)
    print(concat_feat.shape)