from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy

from typing import Union
import librosa
import torch
from src.models.components.xlsr_conformertcm_baseline import Model as XLSRConformerTCM
from src.models.base.adapter_module import AdapterLitModule
from src.utils import load_ln_model_weights
from src.models.components.noise_classifier import FusionNet as NoiseClassifier

def extract_features(wav_np, sr=16000):
    import numpy as np
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    window = 'hamming'
    stft = librosa.stft(wav_np, n_fft=n_fft, hop_length=hop_length, window=window)
    spec_db = librosa.amplitude_to_db(np.abs(stft))
    spec_db = torch.from_numpy(spec_db).float().unsqueeze(0)
    frame_size = int(0.025 * sr)
    hop_size = int(0.010 * sr)
    mfcc = librosa.feature.mfcc(y=wav_np, sr=sr, n_mfcc=13, n_fft=frame_size, hop_length=hop_size, n_mels=n_mels)
    mfcc = torch.from_numpy(mfcc).float().unsqueeze(0)
    f0 = librosa.yin(wav_np, fmin=50, fmax=600, sr=sr, frame_length=n_fft, hop_length=hop_length)
    f0 = np.nan_to_num(f0)
    f0 = torch.from_numpy(f0).float().unsqueeze(0)
    return spec_db, mfcc, f0

class DecisionNoise:
    def __init__(self, model_path:str):
        self.models = {}
        self.model_path = model_path
        self.class_labels = [
        "Clean", "Background Noise", "Background Music", "Gaussian Noise",
        "Bandpass Filter", "Time-Pitch Modulation", "Autotune", "Echo", "Reverberation"]
        self.model = NoiseClassifier(num_classes=len(self.class_labels))
        
        state_dict = torch.load(self.model_path)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        print(f"Loaded noise model from {self.model_path}")
        self.model.eval()
        
    def forward(self, spec, mfcc, f0):
        #spec, mfcc, f0 = spec.to(self.device), mfcc.to(self.device), f0.to(self.device)
        with torch.no_grad():
            out = self.model(spec, mfcc, f0)
        return out
    
    def predict(self, wav_np, sr: int = 16000):
        #print("Type of wav_np = ", type(wav_np))
        spec, mfcc, f0 = extract_features(wav_np, sr)
        spec = spec.unsqueeze(0)
        mfcc = mfcc.unsqueeze(0)
        f0 = f0.unsqueeze(0)
        logits = self.forward(spec, mfcc, f0)
        return logits

    def making_decision(self, wav_np, sr=16000):
        logits = self.predict(wav_np, sr)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()
        pred_label = self.class_labels[pred_idx]
        probs_rounded = [round(float(p), 2) for p in probs]
        return {
            "noise_type": pred_label,
            "probs": probs_rounded
        }

class NormalNCDynamicLoRaLitModule(AdapterLitModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(optimizer, scheduler, args, **kwargs)
        # Initialize metrics
        
        self.decision_nc = DecisionNoise(kwargs.get("noise_model_path", None))
        self.class_labels_to_lora_groups = {
            "Background Noise": "g1",
            "Background Music": "g1",
            #"Gaussian Noise": "g2",
            "Gaussian Noise": "g6",
            "Bandpass Filter": "g3",
            "Time-Pitch Modulation": "g5",
            "Autotune": "g2",
            "Echo": "g4",
            "Reverberation": "g7", # Reverberation is considered as background noise because g1 works better other g2-g6
            "Clean": "g0", # g0 here is not used, it is just for padding
        }
        self.init_criteria(**kwargs)
        self.group = "g0"

    def init_criteria(self, **kwargs) -> torch.nn.Module:
        """
            Initialize the loss function with the given arguments. This method is used to initialize the loss
            function with the given arguments. The loss function is initialized with the given arguments and the
            loss function is returned.
            
            Base model doesn't implement this method. This method should be implemented in the derived
            model class.
        """
        cross_entropy_weight = kwargs.get("cross_entropy_weight", None)
        if cross_entropy_weight is None:
            cross_entropy_weight = torch.tensor(cross_entropy_weight)
        else:
            cross_entropy_weight = torch.tensor([1.0, 1.0])
        self.criterion = torch.nn.CrossEntropyLoss(cross_entropy_weight)
        
        
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def init_adapter(self):
        """Initializes the adapter type.
            This method should be override the parent class.
        """
        is_base_model_path_ln = self.kwargs.get("is_base_model_path_ln", True)
        # Load base model if provided
        if self.base_model_path:
            ckpt = torch.load(self.base_model_path, weights_only=False)
            #print(ckpt)
            #print("is_base_model_path_ln", is_base_model_path_ln)
            if is_base_model_path_ln:
                self.net = load_ln_model_weights(self.net, ckpt['state_dict'])  
            else:
                # Remove the prefix "module." from the keys
                ckpt = {key.replace("module.", ""): value for key, value in ckpt.items()}
                ckpt = {key.replace("_orig_mod.", ""): value for key, value in ckpt.items()}
                self.net.load_state_dict(ckpt)
            print("Loaded baseline model from:", self.base_model_path)

        # Apply adapter method
        if self.use_adapter:
            self.apply_adapter()

        # Load adapters if provided
        if self.adapter_paths:
            self._configure_and_load_adapters()
    def _configure_and_load_adapters(self):
        """Configure and load adapter paths and weights."""
        print(f"Loading adapters from: {self.adapter_paths}")
        
        # Parse adapter paths
        adapter_paths = self.adapter_paths.split(",")
        adapter_names = self.kwargs.get("adapter_names", "")
        adapter_names = adapter_names.split(",")
        
        if len(adapter_paths) != len(adapter_names):
            raise ValueError("adapter_paths and adapter_names must have the same length")
        if len(adapter_paths) == 0:
            raise ValueError("adapter_paths and adapter_names must not be empty")
        if len(adapter_names) == 0:
            raise ValueError("adapter_paths and adapter_names must not be empty")

        # Load adapters
        if len(adapter_paths) > 1:
            print("Loading multiple adapters...")
            #self.load_adapters(adapter_paths, adapter_weights)
            self.net = self.load_multiple_lora_adapters(self.net, adapter_paths, adapter_names)
        else:
            self.load_single_lora_adapter(adapter_paths[0])
            
    def load_single_lora_adapter(self, model, checkpoint_path: str, adapter_name: str = "default"):
        # print(f"Loading LoRA adapter from {checkpoint_path}")
        """Specialized method for loading LoRA adapters"""

        model.load_adapter(checkpoint_path, adapter_name=adapter_name)
        model.set_adapter(adapter_name)

        return model
    
    def load_multiple_lora_adapters(self, model, adapter_paths: list, adapter_names: list):
        for adapter_path, adapter_name in zip(adapter_paths, adapter_names):
            model = self.load_single_lora_adapter(model, adapter_path, adapter_name)
            print(f"Loaded LoRA adapter from {adapter_path} with adapter name {adapter_name}")
        return model
    
    def set_lora_adapter(self, adapter_name: str):
        self.net.set_adapter(adapter_name)
        #print(f"Set LoRA adapter to {adapter_name}")
        
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
            - A dictionary of detailed losses.
        """
        noise_result = self.decision_nc.making_decision(batch[0])
        noise_type = noise_result["noise_type"]
        lora_group = self.route_decision(noise_type)
        self.set_group(lora_group)
        
        if self.group != "g0":
            print(f"set LoRA adapter for model:")
            self.set_lora_adapter(self.group)

        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss
    
    def set_group(self, group: str = "g0"):
        #print(f"Setting group to {group}")
        self.group = group
        
    def route_decision(self, noise_type: str):
        """
        Route decision based on noise type to determine LoRA group
        Returns the LoRA group string for the given noise type
        """
        if noise_type not in self.class_labels_to_lora_groups:
            print(f"Unknown noise type: {noise_type}, defaulting to g0")
            return "g0"
        
        lora_group = self.class_labels_to_lora_groups[noise_type]
        #rint(f"Routed noise type '{noise_type}' to LoRA group '{lora_group}'")
        return lora_group
    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if self.last_emb:
            self._export_embedding_file(batch)
        else:
            if self.score_save_path is not None:
                self._export_score_file(batch, batch_idx)
            else:
                raise ValueError("score_save_path is not provided")

    def _export_score_file(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, inference_mode=True) -> None:
        """Get the score file for the batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        batch_x, utt_id = batch
        
        # Forward pass
        #print(f"batch_x.shape = {batch_x.shape}")
        noise_result = self.decision_nc.making_decision(batch_x[-1].detach().cpu().numpy())
        noise_type = noise_result["noise_type"]
        lora_group = self.route_decision(noise_type)
        self.set_group(lora_group)
        disable_adapter = False
        if self.group != "g0":
            # print(f"set LoRA adapter for model:")
            # self.net.enable_adapters()
            self.set_lora_adapter(self.group)
        else:
            disable_adapter = True
        
        
        if disable_adapter:
            with self.net.disable_adapter():
                batch_out = self.forward(batch_x, inference_mode=inference_mode)
        else:
            batch_out = self.forward(batch_x, inference_mode=inference_mode)
        #preds = torch.argmax(logits, dim=1)
        #return preds, y
    
        #batch_out = self.model_step(batch, inference_mode=inference_mode)
        
        # Optimized tensor to numpy conversion (avoid .data and .tolist())
        if batch_out.is_cuda:
            scores_np = batch_out.detach().cpu().numpy()
        else:
            scores_np = batch_out.detach().numpy()
        
        # Pre-build all lines for batch writing (much faster than line-by-line)
        if self.spec_eval:
            batch_lines = [f'{fname} {scores[0]} {scores[1]}\n' 
                          for fname, scores in zip(utt_id, scores_np)]
        else:
            batch_lines = [f'{fname} {scores[1]}\n' 
                          for fname, scores in zip(utt_id, scores_np)]
        
        # Use buffered writing for maximum performance
        self._write_buffer.extend(batch_lines)
        
        # Flush buffer when it reaches the specified size
        if len(self._write_buffer) >= self._buffer_size * len(batch_lines):
            self._flush_buffer()
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
