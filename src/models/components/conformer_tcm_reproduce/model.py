import torch
import torch.nn as nn
from src.models.components.conformer_tcm_reproduce.conformer import ConformerBlock
from torch.nn.modules.transformer import _get_clones

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)

class MyConformer(nn.Module):
  def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1):
    super(MyConformer, self).__init__()
    print(f'embsize: {emb_size}, heads: {heads}, ffmult: {ffmult}, exp_fac: {exp_fac}, kernel_size: {kernel_size}, n_encoders: {n_encoders}')
    self.dim_head=int(emb_size/heads)
    self.dim=emb_size
    self.heads=heads
    self.kernel_size=kernel_size
    self.n_encoders=n_encoders
    self.positional_emb = nn.Parameter(sinusoidal_embedding(10000, emb_size), requires_grad=False)
    self.encoder_blocks=_get_clones( ConformerBlock( dim = emb_size, dim_head=self.dim_head, heads= heads, 
    ff_mult = ffmult, conv_expansion_factor = exp_fac, conv_kernel_size = kernel_size),
    n_encoders)
    self.class_token = nn.Parameter(torch.rand(1, emb_size))
    self.fc5 = nn.Linear(emb_size, 2)

  def forward(self, x): # x shape [bs, tiempo, frecuencia]
    x = x + self.positional_emb[:, :x.size(1), :]
    x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])#[bs,1+tiempo,emb_size]
    list_attn_weight = []
    for layer in self.encoder_blocks:
            x, attn_weight = layer(x) #[bs,1+tiempo,emb_size]
            list_attn_weight.append(attn_weight)
    embedding=x[:,0,:] #[bs, emb_size]
    out=self.fc5(embedding) #[bs,2]
    return out, embedding