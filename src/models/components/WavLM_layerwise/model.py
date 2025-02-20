from .WavLM import WavLM, WavLMConfig
import torch.nn as nn
import torch
class SSLModelWavlm(nn.Module):
    def __init__(self, cp_path, n_layers):
        super(SSLModelWavlm, self).__init__()
        

        checkpoint = torch.load(cp_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        self.n_layers = n_layers
        self.model = WavLM(cfg)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.encoder.layers = self.model.encoder.layers[:self.n_layers]
        self.out_dim = cfg.encoder_embed_dim
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


# if __name__ == "__main__":
#     from torchinfo import summary
#     model = SSLModelWavlm('/home/hung/WavLM-Large.pt', 12)
#     print(model)