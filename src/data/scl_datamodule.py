#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import librosa
import numpy as np
import os
from scipy import signal
import copy
import random
from src.core_scripts.data_io import wav_augmentation as nii_wav_aug
from src.core_scripts.data_io import wav_tools as nii_wav_tools
from src.data.components.dataio import load_audio, pad
from src.data.components.baseloader import Dataset_base

class Dataset_for(Dataset_base):
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=[],
                    augmentation_methods=[], eval_augment=None, num_additional_real=2, num_additional_spoof=2,
                    trim_length=64000, wav_samp_rate=16000, noise_path=None, rir_path=None,
                    aug_dir=None, online_aug=False, repeat_pad=True, is_train=True):
        
        super(Dataset_for, self).__init__(args, list_IDs, labels, base_dir, algo, vocoders, 
                 augmentation_methods, eval_augment, num_additional_real, num_additional_spoof, 
                 trim_length, wav_samp_rate, noise_path, rir_path, 
                 aug_dir, online_aug, repeat_pad, is_train)
        
        self.vocoded_dir = os.path.join(base_dir, 'vocoded')
        # read spoof_train and spoof_dev list from scp
        if self.is_train:
            self.spoof_list = []
            with open(os.path.join(base_dir, 'scp/spoof_train.lst'), 'r') as f:
                self.spoof_list = f.readlines()
            self.spoof_list = [i.strip() for i in self.spoof_list]
        else:
            self.spoof_list = []
            with open(os.path.join(base_dir, 'scp/spoof_dev.lst'), 'r') as f:
                self.spoof_list = f.readlines()
            self.spoof_list = [i.strip() for i in self.spoof_list]

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        # Anchor real audio sample
        real_audio_file = os.path.join(self.base_dir, self.list_IDs[idx])
        real_audio = load_audio(real_audio_file)

        # Augmented real samples as positive data
        augmented_audios = []
        for augment in self.augmentation_methods:
            augmented_audio = globals()[augment](real_audio, self.args, self.sample_rate, audio_path = real_audio_file)
            # print("aug audio shape",augmented_audio.shape)
            augmented_audios.append(np.expand_dims(augmented_audio, axis=1))


        # Additional real audio samples as positive data
        idxs = list(range(len(self.list_IDs)))
        idxs.remove(idx)  # remove the current audio index
        additional_idxs = np.random.choice(idxs, self.num_additional_real, replace=False)
        additional_audios = [np.expand_dims(load_audio(os.path.join(self.base_dir, self.list_IDs[i])),axis=1) for i in additional_idxs]
        
        # augment the additional real samples
        augmented_additional_audios = []
        for i in range(self.num_additional_real):
            augmethod_index = random.choice(range(len(self.augmentation_methods)))
            tmp = np.expand_dims(globals()[self.augmentation_methods[augmethod_index]](np.squeeze(additional_audios[i],axis=1), self.args, self.sample_rate, 
                                                                                       audio_path = os.path.join(self.base_dir, self.list_IDs[additional_idxs[i]])),axis=1)
            augmented_additional_audios.append(tmp)
            
        
        # Additional spoof audio samples as negative data
        additional_spoof_idxs = np.random.choice(self.spoof_list, self.num_additional_spoof, replace=False)
        additional_spoofs = [np.expand_dims(load_audio(os.path.join(self.base_dir, i)),axis=1) for i in additional_spoof_idxs]
        
        # augment the additional spoof samples
        augmented_additional_spoofs = []
        for i in range(self.num_additional_spoof):
            augmethod_index = random.choice(range(len(self.augmentation_methods)))
            tmp = np.expand_dims(globals()[self.augmentation_methods[augmethod_index]](np.squeeze(additional_spoofs[i],axis=1), self.args, self.sample_rate, audio_path = os.path.join(self.base_dir, additional_spoof_idxs[i])),axis=1)
            augmented_additional_spoofs.append(tmp)
        
        # merge all the data
        batch_data = [np.expand_dims(real_audio, axis=1)] + augmented_audios + additional_audios + augmented_additional_audios + additional_spoofs + augmented_additional_spoofs
        batch_data = nii_wav_aug.batch_pad_for_multiview(
                batch_data, self.sample_rate, self.trim_length, random_trim_nosil=True, repeat_pad=self.repeat_pad)
        batch_data = np.concatenate(batch_data, axis=1)
        # print("batch_data.shape", batch_data.shape)
        
        # return will be anchor ID, batch data and label
        batch_data = Tensor(batch_data)
        # label is 1 for anchor and positive, 0 for vocoded
        label = [1] * (len(augmented_audios) +len(additional_audios) + len(augmented_additional_audios) + 1) + [0] * (len(additional_spoofs) + len(augmented_additional_spoofs))
        # print("label", label)
        return self.list_IDs[idx], batch_data, Tensor(label)			

class Args:
    def __init__(self):
        self.database_path = '/your/path/to/data/ASVspoof_database/DF/'
        self.protocols_path = '/data/hungdx/Datasets/protocols/database/'
        self.batch_size = 14
        self.num_epochs = 100
        self.lr = 0.000001
        self.weight_decay = 0.0001
        self.loss = 'weighted_CCE'
        self.seed = 1234
        self.model_path = None
        self.comment = None
        self.track = 'DF'
        self.eval_output = None
        self.eval = False
        self.is_eval = False
        self.eval_part = 0
        self.cudnn_deterministic_toggle = True
        self.cudnn_benchmark_toggle = False
        self.algo = 3
        self.nBands = 5
        self.minF = 20
        self.maxF = 8000
        self.minBW = 100
        self.maxBW = 1000
        self.minCoeff = 10
        self.maxCoeff = 100
        self.minG = 0
        self.maxG = 0
        self.minBiasLinNonLin = 5
        self.maxBiasLinNonLin = 20
        self.N_f = 5
        self.P = 10
        self.g_sd = 2
        self.SNRmin = 10
        self.SNRmax = 40

class SclNormalDataModule(LightningDataModule):
    """`LightningDataModule` for the ASVSpoof dataset.

    The ASVspoof 2019 database for logical access is based upon a standard multi-speaker speech synthesis database called VCTK2. 
    Genuine speech is collected from 107 speakers (46 male, 61 female) and with no significant channel or background noise effects. 
    Spoofed speech is generated from the genuine data using a number of different spoofing algorithms. 
    The full dataset is partitioned into three subsets, the first for training, the second for development and the third for evaluation.
    There is no speaker overlap across the three subsets regarding target speakers used in voice conversion or TTS adaptation.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        args: Optional[Dict[str, Any]] = None,
        list_IDs: list = [], labels: list = [], base_dir: str = '', algo: int = 5, vocoders: list = [],
        augmentation_methods: list = [], eval_augment: Optional[str] = None, num_additional_real: int = 2, num_additional_spoof: int = 2,
        trim_length: int = 64000, wav_samp_rate: int = 16000, noise_path: Optional[str] = None, rir_path: Optional[str] = None,
        aug_dir: Optional[str] = None, online_aug: bool = False, repeat_pad: bool = True, is_train: bool = True, enable_chunking: bool = False
    ) -> None:
        """Initialize a `ASVSpoofDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.data_dir = data_dir

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of ASVSpoof classes (2).
        """
        return 2

    def prepare_data(self) -> None:        
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            
            # define train dataloader

            d_label_trn, file_train = self.genList(dir_meta=os.path.join(
                    self.data_dir), is_train=True, is_eval=False, is_dev=False)
            
            # if 'portion' in config['data']:
            #     idx = range(len(file_train))
            #     idx = np.random.choice(
            #         idx, int(len(file_train)*config['data']['portion']), replace=False)
            #     file_train = [file_train[i] for i in idx]
            #     if len(d_label_trn) > 0:  # the case of train without label
            #         d_label_trn = {k: d_label_trn[k] for k in file_train}
            
            d_label_dev, file_dev = self.genList(dir_meta=os.path.join(self.data_dir), is_train=False, is_eval=False, is_dev=True)
            
            # if 'portion' in config['data']:
            #     idx = range(len(file_dev))
            #     idx = np.random.choice(
            #         idx, int(len(file_dev)*config['data']['portion']), replace=False)
            #     file_dev = [file_dev[i] for i in idx]
            #     if len(d_label_dev) > 0:  # the case of train without label
            #         d_label_dev = {k: d_label_dev[k] for k in file_dev}
            
            self.data_train = Dataset_for(args, list_IDs=file_train, labels=d_label_trn,
                            base_dir=args.database_path+'/', algo=args.algo, repeat_pad=is_repeat_pad, **config['data']['kwargs'])

            self.data_val = Dataset_for_dev(args, list_IDs=file_dev, labels=d_label_dev,
                                  base_dir=args.database_path+'/', algo=args.algo, repeat_pad=is_repeat_pad, is_train=False, **config['data']['kwargs'])

            self.data_test = Dataset_for_eval(args, list_IDs=file_eval, labels=None,
                                    base_dir=args.database_path+'/', algo=args.algo, repeat_pad=is_repeat_pad, **config['data']['kwargs'],
                                    enable_chunking=config.get(
                                        'eval_chunking', False))
            

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
    
    def genSpoof_list(self, dir_meta, is_train=False, is_eval=False):
        """
        This function is from the following source: https://github.com/TakHemlata/SSL_Anti-spoofing/blob/main/data_utils_SSL.py#L17
        Official source: https://arxiv.org/abs/2202.12233
        Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation
        """
        d_meta = {}
        file_list=[]
        with open(dir_meta, 'r') as f:
            l_meta = f.readlines()

        if (is_train):
            for line in l_meta:
                _,key,_,_,label = line.strip().split()
                
                file_list.append(key)
                d_meta[key] = 1 if label == 'bonafide' else 0
            return d_meta,file_list
        
        elif(is_eval):
            for line in l_meta:
                key= line.strip()
                file_list.append(key)
            return file_list
        else:
            for line in l_meta:
                _,key,_,_,label = line.strip().split()
                
                file_list.append(key)
                d_meta[key] = 1 if label == 'bonafide' else 0
            return d_meta,file_list

if __name__ == "__main__":
    _ = SclNormalDataModule()