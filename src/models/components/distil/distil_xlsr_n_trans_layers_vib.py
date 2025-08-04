import torch.nn as nn
from src.models.components.distil.xlsr_fe import My_XLSR_FE
import torch
import torch.nn as nn


class DropoutForMC(nn.Module):
    """Dropout layer for Bayesian model
    THe difference is that we do dropout even in eval stage
    """

    def __init__(self, p, dropout_flag=True):
        super(DropoutForMC, self).__init__()
        self.p = p
        self.flag = dropout_flag
        return

    def forward(self, x):
        return torch.nn.functional.dropout(x, self.p, training=self.flag)

class BackEndVIB(nn.Module):
    """Back End Wrapper
    """

    def __init__(self, input_dim, out_dim, num_classes,
                 dropout_rate, dropout_flag=True):
        super(BackEndVIB, self).__init__()

        # input feature dimension
        self.in_dim = input_dim
        # output embedding dimension
        self.out_dim = out_dim
        # number of output classes
        self.num_class = num_classes

        # dropout rate
        self.m_mcdp_rate = dropout_rate
        self.m_mcdp_flag = dropout_flag

        # a simple full-connected network for frame-level feature processing
        self.m_frame_level = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),

            nn.Linear(self.in_dim, self.in_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),

            nn.Linear(self.in_dim, self.out_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate)
        )
        # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag))

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
        return logits

class VIB(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, m_mcdp_rate=0.5, mcdp_flag=True):
        super(VIB, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.m_mcdp_rate = m_mcdp_rate
        self.m_mcdp_flag = mcdp_flag

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
        )

        # Latent space
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = self.fc_mu(encoded), self.fc_var(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return z, decoded, mu, logvar

class Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ssl_model = My_XLSR_FE(**kwargs)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.VIB = VIB(128, 128, 64)
        self.backend = BackEndVIB(64, 64, 2, 0.5, False)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)
        x = self.gelu(x)
        x, decoded, mu, logvar = self.VIB(x)
        output = self.backend(x)
        return output