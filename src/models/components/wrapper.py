import torch.nn as nn

class W2V2_TA(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def extract_feat(self, x):
        feat, _ = self.model(x)
        return feat

    def forward(self, x):
        return self.extract_feat(x)