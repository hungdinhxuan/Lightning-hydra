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
from src.data.components.collate_fn import multi_view_collate_fn, variable_multi_view_collate_fn, ChunkingCollator
# augwrapper
from src.data.components.augwrapper import SUPPORTED_AUGMENTATION

# dynamic import of augmentation methods
for aug in SUPPORTED_AUGMENTATION:
    exec(f"from src.data.components.augwrapper import {aug}")


class Dataset_for(Dataset_base):
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=[],
                 augmentation_methods=[], eval_augment=None, num_additional_real=2, num_additional_spoof=2,
                 trim_length=66800, wav_samp_rate=16000, noise_path=None, rir_path=None,
                 aug_dir=None, online_aug=True, repeat_pad=True, is_train=True, random_start=False,
                 **kwargs):
        super(Dataset_for, self).__init__(args, list_IDs, labels, base_dir, algo, vocoders,
                                          augmentation_methods, eval_augment, num_additional_real, num_additional_spoof,
                                          trim_length, wav_samp_rate, noise_path, rir_path,
                                          aug_dir, online_aug, repeat_pad, is_train, random_start)
        # self.args.online_aug = online_aug
        
        # Pre-compute augmentation indices for efficiency
        self.aug_method_count = len(self.augmentation_methods)
        self.use_augmentation = self.aug_method_count > 0

    def __getitem__(self, idx):
        utt_id = self.list_IDs[idx]
        filepath = os.path.join(self.base_dir, utt_id)
        X = load_audio(filepath, self.sample_rate)

        # apply augmentation - optimized selection
        if self.use_augmentation:
            # More efficient than random.choice(range(len(...)))
            augmethod_index = random.randrange(self.aug_method_count)
            # print("Augmenting with", self.augmentation_methods[augmethod_index])
            X = globals()[self.augmentation_methods[augmethod_index]](X, self.args, self.sample_rate,
                                                                      audio_path=filepath)

        x_inp = Tensor(X)
        target = self.labels[utt_id]
        return x_inp, target


class Dataset_for_dev(Dataset_base):
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=[],
                 augmentation_methods=[], eval_augment=None, num_additional_real=2, num_additional_spoof=2,
                 trim_length=66800, wav_samp_rate=16000, noise_path=None, rir_path=None,
                 aug_dir=None, online_aug=True, repeat_pad=False, is_train=True,
                 random_start=False, **kwargs
                 ):
        super(Dataset_for_dev, self).__init__(args, list_IDs, labels, base_dir, algo, vocoders,
                                              augmentation_methods, eval_augment, num_additional_real, num_additional_spoof,
                                              trim_length, wav_samp_rate, noise_path, rir_path,
                                              aug_dir, online_aug, repeat_pad, is_train, random_start)
        self.is_dev_aug = kwargs.get('is_dev_aug', False)
        # Use consistent caching
        self.cache_dir = kwargs.get('cache_dir')
        
        # Pre-compute augmentation indices for efficiency
        self.aug_method_count = len(self.augmentation_methods)
        self.use_augmentation = self.is_dev_aug and self.aug_method_count > 0
        
        if self.is_dev_aug:
            print("Dev aug is enabled")

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        filepath = os.path.join(self.base_dir, utt_id)
        
        # Use load_audio for consistency and caching
        X = load_audio(filepath, sr=self.sample_rate, cache_dir=self.cache_dir)
        
        # Optimized augmentation selection
        if self.use_augmentation:
            # More efficient than random.choice(range(len(...)))
            augmethod_index = random.randrange(self.aug_method_count)
            # print("Augmenting with", self.augmentation_methods[augmethod_index])
            X = globals()[self.augmentation_methods[augmethod_index]](X, self.args, self.sample_rate,
                                                                    audio_path=filepath)
        x_inp = Tensor(X)
        target = self.labels[utt_id]
        return x_inp, target


class Dataset_for_eval(Dataset_base):
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=[],
                 augmentation_methods=[], eval_augment=None, num_additional_real=2, num_additional_spoof=2,
                 trim_length=66800, wav_samp_rate=16000, noise_path=None, rir_path=None,
                 aug_dir=None, online_aug=True, repeat_pad=True, is_train=True, enable_chunking=False, random_start=False, **kwargs
                 ):
        super(Dataset_for_eval, self).__init__(args, list_IDs, labels, base_dir, algo, vocoders,
                                               augmentation_methods, eval_augment, num_additional_real, num_additional_spoof,
                                               trim_length, wav_samp_rate, noise_path, rir_path,
                                               aug_dir, online_aug, repeat_pad, is_train, random_start)
        self.enable_chunking = enable_chunking
        self.padding_type = "repeat" if repeat_pad else "zero"
        # Use consistent caching
        self.cache_dir = kwargs.get('cache_dir')
        # print("Chunking enabled:", self.enable_chunking)
        # print("trim_length:", trim_length)
        # print("padding_type:", self.padding_type)
        self.no_pad = args.get('no_pad', False) if args is not None else False
        if self.no_pad:
            print('No padding')

    def __getitem__(self, idx):
        utt_id = self.list_IDs[idx]
        # print("utt_id:", utt_id)
        # print("self.base_dir:", self.base_dir)
        filepath = os.path.join(self.base_dir, utt_id)
        #print("filepath:", filepath)
        
        # Use load_audio for consistency and caching
        X = load_audio(filepath, sr=self.sample_rate, cache_dir=self.cache_dir)
        
        #print("loaded X:", X.shape)
        # apply augmentation at inference time
        if self.eval_augment is not None:
            # print("eval_augment:", self.eval_augment)
            X = globals()[self.eval_augment](
                X, self.args, self.sample_rate, audio_path=filepath)
        if not self.enable_chunking and not self.no_pad:
            X = pad(X, padding_type=self.padding_type,
                    max_len=self.trim_length, random_start=self.random_start)
        if self.no_pad and len(X) > 160000: # 10 seconds is the maximum
            X = X[:160000]
        x_inp = Tensor(X)
        return x_inp, utt_id


class NormalDataModule(LightningDataModule):
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
        chunking_eval: bool = False,
        enable_cache: bool = False,
    ) -> None:
        """Initialize a `ASVSpoofDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param enable_cache: Whether to enable caching. Defaults to `False`.
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
        self.args = args
        self.enable_cache = enable_cache
        self.protocol_path = args.get(
            'protocol_path', os.path.join(self.data_dir, 'protocol.txt'))
        self.is_variable_multi_view = args.get(
            'is_variable_multi_view', False) if args is not None else False
        if self.is_variable_multi_view:
            print('Using variable multi-view collate function')
            self.top_k = self.args.get('top_k', 4)
            self.min_duration = self.args.get('min_duration', 16000)
            self.max_duration = self.args.get('max_duration', 64000)
            self.collate_fn = lambda x: variable_multi_view_collate_fn(
                x,
                self.top_k,
                self.min_duration,
                self.max_duration,
                self.args.wav_samp_rate,
                self.args.padding_type,
                self.args.random_start
            )
        else:
            self.collate_fn = lambda x: multi_view_collate_fn(
                x,
                self.args.views,
                self.args.wav_samp_rate,
                self.args.padding_type,
                self.args.random_start,
                self.args.view_padding_configs
            )
        self.chunking_eval = False
        if chunking_eval:
            self.chunking_eval = True
            collator_params: Dict[str, Any] = {
                "chunk_size": self.args.get('chunk_size', 16000),  # 1 second
                # 0.5 second
                "overlap_size": self.args.get('overlap_size', 8000),
                "enable_chunking": True
            }
            self.eval_collator = ChunkingCollator(
                **collator_params)
        else:
            self.eval_collator = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of ASVSpoof classes (2).
        """
        return 2

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
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
            d_label_trn, file_train = self.genList(
                is_train=True, is_eval=False, is_dev=False)

            print('no. of training trials', len(file_train))

            d_label_dev, file_dev = self.genList(
                is_train=False, is_eval=False, is_dev=True)
            print('no. of validation trials', len(file_dev))
            d_meta, file_eval = self.genList(
                is_train=False, is_eval=True, is_dev=False)
            print('no. of evaluation trials', len(file_eval))

            # Add cache settings to args
            if self.args is None:
                self.args = {}
            
            # Check both enable_cache parameter and args.enable_cache
            cache_enabled = self.enable_cache or self.args.get('enable_cache', False)
            cache_dir = self.args.get('cache_dir')
            
            if cache_enabled and cache_dir is not None:
                print(f"Cache is ENABLED")
                print(f"Using cache directory: {cache_dir}")
                self.args['cache_dir'] = cache_dir
            else:
                print(f"Cache is DISABLED")
                self.args['cache_dir'] = None

            self.data_train = Dataset_for(self.args, list_IDs=file_train, labels=d_label_trn,
                                          base_dir=self.data_dir+'/',  **self.args)

            self.data_val = Dataset_for_dev(self.args, list_IDs=file_dev, labels=d_label_dev,
                                            base_dir=self.data_dir+'/',  is_train=False, **self.args)
            self.no_pad = self.args.get('no_pad', False) if self.args is not None else False

            self.data_test = Dataset_for_eval(self.args, list_IDs=file_eval, labels=None,
                                              base_dir=self.data_dir+'/',  random_start=self.args.random_start, trim_length=self.args.trim_length, repeat_pad=True if self.args.padding_type == 'repeat' else False,
                                              enable_chunking=self.chunking_eval)

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
            drop_last=True,
            collate_fn=self.collate_fn,
            persistent_workers=True
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
            collate_fn=self.collate_fn,
            persistent_workers=True if self.hparams.num_workers > 0 else False
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device if self.no_pad is False else 1, # if no_pad is True, we need to set batch_size to 1
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.eval_collator,
            persistent_workers=True if self.hparams.num_workers > 0 else False
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

    def genList(self, is_train=False, is_eval=False, is_dev=False):
        """
            This function generates the list of files and their corresponding labels
            Specifically for the standard CNSL dataset
            Optimized to read protocol file only once
        """
        # bonafide: 1, spoof: 0
        d_meta = {}
        file_list = []

        # Read protocol file only once - major optimization!
        with open(self.protocol_path, 'r') as f:
            l_meta = f.readlines()
        
        # Parse all data in single pass
        for line in l_meta:
            utt, subset, label = line.strip().split()
            label_val = 1 if label == 'bonafide' else 0
            
            if is_train and subset == 'train':
                file_list.append(utt)
                d_meta[utt] = label_val
            elif is_dev and subset == 'dev':
                file_list.append(utt)
                d_meta[utt] = label_val
            elif is_eval and (subset == 'eval' or subset == 'test'):
                file_list.append(utt)
                d_meta[utt] = label_val

        return d_meta, file_list


if __name__ == "__main__":
    _ = NormalDataModule()
