#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple, List
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler
from torch import Tensor
import librosa
import numpy as np
import os
import random
import math
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


class ReplayDataset(Dataset_base):
    def __init__(self, args, novel_list_IDs, novel_labels, replay_list_IDs, replay_labels, 
                 base_dir, novel_ratio=0.7, replay_ratio=0.3, algo=5, vocoders=[],
                 augmentation_methods=[], eval_augment=None, num_additional_real=2, num_additional_spoof=2,
                 trim_length=66800, wav_samp_rate=16000, noise_path=None, rir_path=None,
                 aug_dir=None, online_aug=True, repeat_pad=True, is_train=True, random_start=False,
                 **kwargs):
        super(ReplayDataset, self).__init__(args, novel_list_IDs, novel_labels, base_dir, algo, vocoders,
                                          augmentation_methods, eval_augment, num_additional_real, num_additional_spoof,
                                          trim_length, wav_samp_rate, noise_path, rir_path,
                                          aug_dir, online_aug, repeat_pad, is_train, random_start)
        
        self.novel_list_IDs = novel_list_IDs
        self.novel_labels = novel_labels
        self.replay_list_IDs = replay_list_IDs
        self.replay_labels = replay_labels
        self.novel_ratio = novel_ratio
        self.replay_ratio = replay_ratio
        
        # Add cache support
        self.cache_dir = kwargs.get('cache_dir')
        
        # Pre-compute augmentation indices for efficiency
        self.aug_method_count = len(self.augmentation_methods)
        self.use_augmentation = self.aug_method_count > 0
        
        # Validate ratios
        if novel_ratio + replay_ratio > 1.0:
            raise ValueError(f"Sum of ratios ({novel_ratio + replay_ratio}) cannot exceed 1.0")
        
        # Calculate dataset sizes based on novel set (novel set determines epoch length)
        self.novel_size = len(novel_list_IDs)
        self.replay_size = len(replay_list_IDs)
        
        print(f"Novel set size: {self.novel_size}")
        print(f"Replay set size: {self.replay_size}")
        print(f"Novel ratio: {novel_ratio}, Replay ratio: {replay_ratio}")

    def __len__(self):
        # Return length based on novel set
        return self.novel_size

    def _get_sample(self, utt_id, labels_dict):
        """Unified method to get a sample (removes code duplication)"""
        filepath = os.path.join(self.base_dir, utt_id)
        X = load_audio(filepath, sr=self.sample_rate, cache_dir=self.cache_dir)

        # Optimized augmentation selection
        if self.use_augmentation:
            augmethod_index = random.randrange(self.aug_method_count)
            X = globals()[self.augmentation_methods[augmethod_index]](X, self.args, self.sample_rate,
                                                                      audio_path=filepath)

        x_inp = Tensor(X)
        target = labels_dict[utt_id]
        return x_inp, target

    def __getitem__(self, idx):
        # Handle batch indices from custom sampler
        if isinstance(idx, list):
            # This is a list of (data_type, index) tuples from our custom sampler
            batch_data = []
            for data_type, sample_idx in idx:
                if data_type == 'novel':
                    sample = self.get_novel_sample(sample_idx)
                else:  # replay
                    sample = self.get_replay_sample(sample_idx)
                batch_data.append(sample)
            return batch_data
        else:
            # Single index - fallback to novel samples
            utt_id = self.novel_list_IDs[idx]
            return self._get_sample(utt_id, self.novel_labels)

    def get_novel_sample(self, idx):
        """Get a sample from the novel set"""
        utt_id = self.novel_list_IDs[idx]
        return self._get_sample(utt_id, self.novel_labels)

    def get_replay_sample(self, idx=None):
        """Get a random sample from the replay set"""
        if idx is None:
            idx = random.randint(0, self.replay_size - 1)
        else:
            idx = idx % self.replay_size  # Handle overflow
            
        utt_id = self.replay_list_IDs[idx]
        return self._get_sample(utt_id, self.replay_labels)


class ReplaySampler(Sampler):
    """Optimized sampler that handles novel and replay data mixing"""
    
    def __init__(self, dataset: ReplayDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.novel_ratio = dataset.novel_ratio
        self.replay_ratio = dataset.replay_ratio
        
        # Calculate samples per batch
        self.novel_per_batch = int(batch_size * self.novel_ratio)
        self.replay_per_batch = int(batch_size * self.replay_ratio)
        
        # Adjust if needed to not exceed batch_size
        total_per_batch = self.novel_per_batch + self.replay_per_batch
        if total_per_batch > batch_size:
            self.novel_per_batch = batch_size - self.replay_per_batch
        
        self.total_per_batch = self.novel_per_batch + self.replay_per_batch
        
        # Pre-generate replay indices for efficiency
        self.replay_size = len(dataset.replay_list_IDs)
        
        print(f"Batch composition: {self.novel_per_batch} novel + {self.replay_per_batch} replay = {self.total_per_batch} total")

    def __iter__(self):
        # Generate novel indices once
        novel_indices = list(range(len(self.dataset.novel_list_IDs)))
        if self.shuffle:
            random.shuffle(novel_indices)
        
        # Pre-generate enough replay indices for entire epoch
        num_batches = math.ceil(len(novel_indices) / self.novel_per_batch)
        total_replay_needed = num_batches * self.replay_per_batch
        replay_indices = [random.randrange(self.replay_size) for _ in range(total_replay_needed)]
        
        replay_idx = 0
        
        # Generate batches more efficiently
        for i in range(0, len(novel_indices), self.novel_per_batch):
            batch_indices = []
            
            # Add novel samples
            novel_batch = novel_indices[i:i + self.novel_per_batch]
            batch_indices.extend([('novel', idx) for idx in novel_batch])
            
            # Add replay samples (pre-generated)
            replay_batch = replay_indices[replay_idx:replay_idx + self.replay_per_batch]
            batch_indices.extend([('replay', idx) for idx in replay_batch])
            replay_idx += self.replay_per_batch
            
            # Shuffle the batch if needed
            if self.shuffle:
                random.shuffle(batch_indices)
            
            yield batch_indices

    def __len__(self):
        return math.ceil(len(self.dataset.novel_list_IDs) / self.novel_per_batch)



class ReplayDataLoader(DataLoader):
    """Simplified DataLoader for replay dataset"""
    
    def __init__(self, dataset: ReplayDataset, batch_size: int, shuffle: bool = True, 
                 num_workers: int = 0, pin_memory: bool = False, collate_fn=None, 
                 persistent_workers: bool = False, **kwargs):
        # Use custom sampler
        sampler = ReplaySampler(dataset, batch_size, shuffle)
        
        super().__init__(
            dataset=dataset,
            batch_size=None,  # We handle batching in the sampler
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn or self._default_collate_fn,
            persistent_workers=persistent_workers,
            **kwargs
        )

    def _default_collate_fn(self, batch):
        """Simplified default collate function"""
        # batch is a list containing one item (the complete batch from __getitem__)
        if len(batch) == 1 and isinstance(batch[0], list):
            # Extract the actual batch data
            batch_data = batch[0]
        else:
            batch_data = batch
        
        # Simple default collate - separate samples and targets
        samples, targets = zip(*batch_data)
        return list(samples), list(targets)


class ReplayDataModule(LightningDataModule):
    """LightningDataModule for replay learning with novel and replay sets"""

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        args: Optional[Dict[str, Any]] = None,
        novel_ratio: float = 0.7,
        replay_ratio: float = 0.3,
        novel_protocol_path: Optional[str] = None,
        replay_protocol_path: Optional[str] = None,
        chunking_eval: bool = False,
        enable_cache: bool = False,
    ) -> None:
        """Initialize a ReplayDataModule.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param args: Additional arguments.
        :param novel_ratio: Ratio of novel samples in each batch. Defaults to `0.7`.
        :param replay_ratio: Ratio of replay samples in each batch. Defaults to `0.3`.
        :param novel_protocol_path: Path to novel set protocol file.
        :param replay_protocol_path: Path to replay set protocol file.
        :param chunking_eval: Whether to use chunking for evaluation. Defaults to `False`.
        :param enable_cache: Whether to enable caching. Defaults to `False`.
        """
        super().__init__()

        # Validate ratios
        if novel_ratio + replay_ratio > 1.0:
            raise ValueError(f"Sum of ratios ({novel_ratio + replay_ratio}) cannot exceed 1.0")

        # Save hyperparameters
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.data_dir = data_dir
        self.args = args
        self.enable_cache = enable_cache
        self.novel_ratio = novel_ratio
        self.replay_ratio = replay_ratio
        
        # Set protocol paths
        self.novel_protocol_path = novel_protocol_path or os.path.join(self.data_dir, 'novel_protocol.txt')
        self.replay_protocol_path = replay_protocol_path or os.path.join(self.data_dir, 'replay_protocol.txt')
        
        # Setup collate function
        self.is_variable_multi_view = args.get('is_variable_multi_view', False) if args is not None else False
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
        
        # Setup evaluation collator
        self.chunking_eval = chunking_eval
        if chunking_eval:
            collator_params: Dict[str, Any] = {
                "chunk_size": self.args.get('chunk_size', 16000),
                "overlap_size": self.args.get('overlap_size', 8000),
                "enable_chunking": True
            }
            self.eval_collator = ChunkingCollator(**collator_params)
        else:
            self.eval_collator = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return 2

    def prepare_data(self) -> None:
        """Download data if needed. This method is called only from a single GPU."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        # Divide batch size by the number of devices
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Load datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # Setup cache configuration first
            if self.args is None:
                self.args = {}
            
            cache_enabled = self.enable_cache or self.args.get('enable_cache', False)
            cache_dir = self.args.get('cache_dir')
            
            if cache_enabled and cache_dir is not None:
                print(f"Cache is ENABLED")
                print(f"Using cache directory: {cache_dir}")
                self.args['cache_dir'] = cache_dir
            else:
                print(f"Cache is DISABLED")
                self.args['cache_dir'] = None
            
            # Load novel and replay sets efficiently
            print("Loading novel dataset...")
            novel_labels_trn, novel_files_trn = self.genList(self.novel_protocol_path, is_train=True)
            novel_labels_dev, novel_files_dev = self.genList(self.novel_protocol_path, is_dev=True)
            novel_labels_eval, novel_files_eval = self.genList(self.novel_protocol_path, is_eval=True)
            
            print("Loading replay dataset...")
            replay_labels_trn, replay_files_trn = self.genList(self.replay_protocol_path, is_train=True)
            replay_labels_dev, replay_files_dev = self.genList(self.replay_protocol_path, is_dev=True)
            replay_labels_eval, replay_files_eval = self.genList(self.replay_protocol_path, is_eval=True)
            
            print(f'Novel training trials: {len(novel_files_trn)}')
            print(f'Novel validation trials: {len(novel_files_dev)}')
            print(f'Novel evaluation trials: {len(novel_files_eval)}')
            print(f'Replay training trials: {len(replay_files_trn)}')
            print(f'Replay validation trials: {len(replay_files_dev)}')
            print(f'Replay evaluation trials: {len(replay_files_eval)}')

            # Create datasets
            self.data_train = ReplayDataset(
                self.args, 
                novel_list_IDs=novel_files_trn, 
                novel_labels=novel_labels_trn,
                replay_list_IDs=replay_files_trn, 
                replay_labels=replay_labels_trn,
                base_dir=self.data_dir+'/',
                novel_ratio=self.novel_ratio,
                replay_ratio=self.replay_ratio,
                **self.args
            )

            # For validation and test, we can use the optimized datasets from normal_multiview_datamodule
            from src.data.normal_multiview_datamodule import Dataset_for_dev, Dataset_for_eval
            
            self.data_val = Dataset_for_dev(
                self.args, 
                list_IDs=novel_files_dev, 
                labels=novel_labels_dev,
                base_dir=self.data_dir+'/', 
                is_train=False, 
                **self.args
            )

            self.data_test = Dataset_for_eval(
                self.args, 
                list_IDs=novel_files_eval, 
                labels=None,
                base_dir=self.data_dir+'/', 
                random_start=self.args.random_start, 
                trim_length=self.args.trim_length, 
                repeat_pad=True if self.args.padding_type == 'repeat' else False,
                enable_chunking=self.chunking_eval
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return ReplayDataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
            persistent_workers=True if self.hparams.num_workers > 0 else False
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
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
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.eval_collator,
            persistent_workers=True if self.hparams.num_workers > 0 else False
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after training/testing."""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint."""
        pass

    def genList(self, protocol_path: str, is_train=False, is_eval=False, is_dev=False):
        """Generate list of files and their corresponding labels from protocol file.
        Optimized to read protocol file only once.
        """
        d_meta = {}
        file_list = []

        if not os.path.exists(protocol_path):
            raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

        # Read protocol file only once - major optimization!
        with open(protocol_path, 'r') as f:
            l_meta = f.readlines()

        # Parse all data in single pass
        for line in l_meta:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            utt, subset, label = parts[:3]
            label_val = 1 if label == 'bonafide' else 0
            
            if (is_train and subset == 'train') or \
               (is_dev and subset == 'dev') or \
               (is_eval and subset in ['eval', 'test']):
                file_list.append(utt)
                d_meta[utt] = label_val

        return d_meta, file_list
