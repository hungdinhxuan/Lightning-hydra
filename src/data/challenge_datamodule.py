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

# augwrapper
from src.data.components.augwrapper import SUPPORTED_AUGMENTATION

# dynamic import of augmentation methods
for aug in SUPPORTED_AUGMENTATION:
    exec(f"from src.data.components.augwrapper import {aug}")

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
        self.cache_dir = args.get('cache_dir', None)
        # print("Chunking enabled:", self.enable_chunking)
        print("trim_length:", trim_length)
        # print("padding_type:", self.padding_type)
        # print("random_start:", random_start)
        self.no_pad = args.get('no_pad', False) if args is not None else False
        if self.no_pad:
            print('No padding')

    def __getitem__(self, idx):
        utt_id = self.list_IDs[idx]
        filepath = os.path.join(self.base_dir, utt_id)
        X = load_audio(filepath, self.sample_rate, cache_dir=self.cache_dir)
        
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


class ChallengeDataModule(LightningDataModule):
    
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
        self.chunking_eval = chunking_eval
        self.data_dir = data_dir
        self.protocol_path = args.get(
            'protocol_path', os.path.join(self.data_dir, 'protocol.txt'))

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of ASVSpoof classes (2).
        """
        return 2

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

            file_eval = self.genList(
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
            self.data_test = Dataset_for_eval(self.args, list_IDs=file_eval, labels=None,
                                              base_dir=self.data_dir+'/',  random_start=self.args.random_start, trim_length=self.args.trim_length, repeat_pad=True if self.args.padding_type == 'repeat' else False,
                                              enable_chunking=self.chunking_eval)

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

    def genList(self, seperate_by: str = " "):
        """
            This function generates the list of files and their corresponding labels
            Specifically for the standard CNSL dataset
        """

        file_list = []

            # no eval self.protocol_path yet
        with open(self.protocol_path, 'r') as f:
            l_meta = f.readlines()
        for line in l_meta:
            parts = line.strip().split(seperate_by)
            file_list.append(parts[0]) # split by space, the first part is the file path

        return file_list


if __name__ == "__main__":
    _ = ChallengeDataModule()
