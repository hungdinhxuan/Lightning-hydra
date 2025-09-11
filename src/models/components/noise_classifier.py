import torch
import torch.nn as nn
import torch.nn.functional as F

# 최적화된 Spectrogram Branch
class SpectrogramBranchLSTM(nn.Module):
    def __init__(self, in_channels=1, output_dim=1024, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden*2, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):         # x: [B, 1, F, T]
        x = self.cnn(x)          # [B, 256, F', T']
        x = x.mean(2)            # freq average pooling [B, 256, T']
        x = x.permute(0, 2, 1)   # [B, T', 256]
        lstm_out, _ = self.lstm(x) # [B, T', lstm_hidden*2]
        x = lstm_out.mean(dim=1) # [B, lstm_hidden*2]
        x = self.fc(x)
        return x

# MFCC Branch (CNN+LSTM)
class MFCCBranchLSTM(nn.Module):
    def __init__(self, in_channels=1, output_dim=1024, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden*2, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):         # x: [B, 1, F, T]
        x = self.cnn(x)          # [B, 256, F', T']
        x = x.mean(2)            # freq average pooling [B, 256, T']
        x = x.permute(0, 2, 1)   # [B, T', 256]
        lstm_out, _ = self.lstm(x) # [B, T', lstm_hidden*2]
        x = lstm_out.mean(dim=1) # [B, lstm_hidden*2]
        x = self.fc(x)
        return x


# F0 Branch (동일하게 유지)
class F0Branch(nn.Module):
    def __init__(self, input_len, output_dim=1024, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden*2, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [batch, 1, f0_len]
        x = self.conv(x)           # [batch, 64, T]
        x = x.permute(0, 2, 1)     # [batch, T, 64]
        lstm_out, _ = self.lstm(x) # [batch, T, lstm_hidden*2]
        x = lstm_out.mean(dim=1)   # [batch, lstm_hidden*2] (global average pooling)
        x = self.fc(x)             # [batch, output_dim]
        return x

# FusionNet (branch 교체)
class FusionNet(nn.Module):
    def __init__(self, num_classes, branch_output_dim=1024, spec_shape=(1,1025,126), mfcc_shape=(1,13,401), f0_len=126):
        super().__init__()
        self.spec_branch = SpectrogramBranchLSTM(in_channels=spec_shape[0], output_dim=branch_output_dim)
        self.mfcc_branch = MFCCBranchLSTM(in_channels=mfcc_shape[0], output_dim=branch_output_dim)
        self.f0_branch = F0Branch(f0_len, output_dim=branch_output_dim)  # CNN+LSTM

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(branch_output_dim * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, spec, mfcc, f0):
        spec_feat = self.spec_branch(spec)
        mfcc_feat = self.mfcc_branch(mfcc)
        f0_feat = self.f0_branch(f0)
        fused = torch.cat([spec_feat, mfcc_feat, f0_feat], dim=1)
        out = self.classifier(fused)
        return out
