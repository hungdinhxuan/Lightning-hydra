import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from transformers import Wav2Vec2Model
import csv
from tqdm import tqdm
from pooling import MultiHeadAttentionPooling
from multiconv_cgmlp import MultiConvolutionalGatingMLP


def hsic_unbiased(K, L):
    """
    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC) as per Equation 5 in the paper.
    > Reference: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    """
    m = K.shape[0]

    # Zero out the diagonal elements of K and L
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def hsic_biased(K, L):
    """ Compute the biased HSIC (the original CKA) """
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)

def cka(feats_A, feats_B, kernel_metric='ip', rbf_sigma=1.0, unbiased=False):
        """Computes the unbiased Centered Kernel Alignment (CKA) between features."""
        
        if kernel_metric == 'ip':
            # Compute kernel matrices for the linear case
            K = torch.mm(feats_A, feats_A.T)
            L = torch.mm(feats_B, feats_B.T)
        elif kernel_metric == 'rbf':
            # COMPUTES RBF KERNEL
            K = torch.exp(-torch.cdist(feats_A, feats_A) ** 2 / (2 * rbf_sigma ** 2))
            L = torch.exp(-torch.cdist(feats_B, feats_B) ** 2 / (2 * rbf_sigma ** 2))
        else:
            raise ValueError(f"Invalid kernel metric {kernel_metric}")

        # Compute HSIC values
        hsic_fn = hsic_unbiased if unbiased else hsic_biased
        hsic_kk = hsic_fn(K, K)
        hsic_ll = hsic_fn(L, L)
        hsic_kl = hsic_fn(K, L)

        # Compute CKA
        #print('hsic', hsic_kl)
        cka_value = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)        
        return cka_value.item()

class SwiGLU(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_1 = nn.Linear(dimension,dimension)
        self.linear_2 = nn.Linear(dimension,dimension)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.linear_2(x)

        return swiglu


class SSLClassifier(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.feature_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
        
        self.blocks = nn.ModuleList()
        for _ in range(4):
            self.blocks.append(
                nn.Sequential(
                    MultiConvolutionalGatingMLP(size=128,
                                                 linear_units=1024,
                                                 arch_type="concat_fusion",
                                                 kernel_sizes="3,7,11,15",
                                                 merge_conv_kernel=15,
                                                 use_non_linear=True,
                                                 dropout_rate=0.1,
                                                 use_linear_after_conv=True,
                                                 activation="silu",
                                                 gate_activation="silu"
                                                 )
                )
            )
            
        self.feature_projection = nn.Linear(1024, 128)
        self.pooling = MultiHeadAttentionPooling(512)
        self.silu = SwiGLU(128)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SELU(inplace=True),
            nn.Linear(512, 2)
        )
        self.config = config
        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]))
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        _, self.d_meta = self._get_list_IDs(self.config['evaluation']['protocol_path'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x, output_hidden_states=True)
        hidden_states = torch.stack(x.hidden_states, dim=1)
        hidden_states = self.feature_projection(hidden_states)
        hidden_states = self.silu(hidden_states)
        x = torch.sum(hidden_states, 1)
        hidden_states_processed = []
        for i, b in enumerate(self.blocks):
            x = b(x)
            hidden_states_processed.append(x)
        hidden_states_processed = torch.stack(hidden_states_processed, dim=1)
        hidden_states_processed = hidden_states_processed.view(hidden_states_processed.shape[0], hidden_states_processed.shape[2], -1)  # Shape: [batch_size, seq_len, num_layers * proj_dim]
        x = self.pooling(hidden_states_processed.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.classifier(x.squeeze(1))
        return x

    def on_epoch_end(self, outputs, phase):
        all_scores = []
        all_labels = []
        all_losses = []
        
        for preds, labels, loss in outputs:
            all_scores.append(preds)
            all_labels.append(labels)
            all_losses.append(loss)
        
        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)
        all_losses = torch.stack(all_losses)
        
        all_scores = F.softmax(all_scores, dim=-1)
        self.accuracy(torch.argmax(all_scores, 1), all_labels)
        
        self.log_dict(
            {f"{phase}_loss": all_losses.mean(),
             f"{phase}_accuracy": self.accuracy},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.feature_extractor(x, output_hidden_states=True)
        hidden_states = torch.stack(x.hidden_states, dim=1)
        hidden_states = self.feature_projection(hidden_states)
        hidden_states = self.silu(hidden_states)
        x = torch.sum(hidden_states, 1)
        hidden_states_processed = []
        for i, b in enumerate(self.blocks):
            x = b(x)
            hidden_states_processed.append(x)
        hidden_states_processed = torch.stack(hidden_states_processed, dim=1)
        hidden_states_processed = hidden_states_processed.view(hidden_states_processed.shape[0], hidden_states_processed.shape[2], -1)  # Shape: [batch_size, seq_len, num_layers * proj_dim]
        x = self.pooling(hidden_states_processed.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
        mean, std = torch.split(x, 512, dim=-1)
        x = self.classifier(x)
        cross_entropy_loss = self.loss_fn(x, y)
        scores = self.accuracy(torch.argmax(x, 1), y)
        cls = mean.reshape(mean.shape[0], 4, -1)
        indices = torch.triu_indices(4, 4, offset=1)
        cka_values = []
        for i, j in zip(indices[0], indices[1]):
            cka_values.append(cka(cls[:, i, :], cls[:, j, :]))
        cka_values = torch.tensor(cka_values)
        cka_loss = torch.mean(torch.abs(cka_values))
        loss = cross_entropy_loss + cka_loss

        self.training_step_outputs.append((scores, y, loss))

        return loss

    def on_train_epoch_end(self):
        self.on_epoch_end(self.training_step_outputs, phase="train")
        self.training_step_outputs.clear()
        self.accuracy.reset()


    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.validation_step_outputs.append((scores, y, loss))

        return loss
    
    def on_validation_epoch_end(self):
        self.on_epoch_end(self.validation_step_outputs, phase="val")
        self.validation_step_outputs.clear()
        self.accuracy.reset()

    def test_step(self, batch, batch_idx):
        self._produce_evaluation_file(batch, batch_idx)

    def _get_list_IDs(self, input_file):
        delimiter = ','
        list_IDs = []
        d_meta = {}
        with open(input_file, 'r') as infile:
            reader = csv.reader(infile, delimiter=delimiter)
            first_row = next(reader)
            if 'file_name' in first_row or 'label' in first_row:
                print("Skipping header row in the input file.")
            else:
                reader = [first_row] + list(reader)
            total_lines = sum(1 for _ in open(input_file)) - 1
            infile.seek(0)
            for row in tqdm(reader, total=total_lines, desc=f"Processing Ids"):
                try:
                    file_name, label = row
                    d_meta[file_name] = 1 if label == "bonafide" else 0
                    list_IDs.append(file_name)
                except ValueError as e:
                    print(f"Skipping malformed row: {row}. Error: {e}")

        return list_IDs, d_meta

    def _produce_evaluation_file(self, batch, batch_idx):
        x, utt_id = batch
        fname_list = []
        score_list = []
        out = self(x)
        out = F.log_softmax(out, dim=-1)
        ss = out[:, 0]
        bs = out[:, 1]
        llr = bs - ss
        if self.config['evaluation']['task'] == "asvspoof":
            utt_id = tuple(item.split('/')[-1].split('.')[0] for item in utt_id)
        fname_list.extend(utt_id)
        score_list.extend(llr.tolist())
            
        with open(self.config['evaluation']['output_score_file'], "a+") as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write("{} {}\n".format(f, cm))
        fh.close()

    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                self.parameters(), lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
        )

        return optimizer