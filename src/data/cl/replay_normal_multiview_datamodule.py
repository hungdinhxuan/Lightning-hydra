#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple, Union, List
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import librosa
import numpy as np
import os
from scipy import signal
import copy
import random


import multiprocessing
import pickle
from functools import partial
from tqdm import tqdm

def _group_samples(dataset, class_id):
    samples = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        if y == class_id:
            samples.append((x, y, i))
    return samples

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

class EmptyDataset(Dataset):
    """Empty dataset class for skipping training/validation"""
    def __init__(self):
        super().__init__()
        
    def __len__(self):
        return 0
        
    def __getitem__(self, idx):
        return None

class ReplayDataset(Dataset):
    """Dataset that combines current task samples with memory buffer samples."""
    
    def __init__(self, current_dataset, memory_buffer):
        """
        Initialize the replay dataset.
        
        Args:
            current_dataset: Dataset for the current task
            memory_buffer: Dictionary of stored samples {class_id: [(x, y), ...]}
        """
        self.current_dataset = current_dataset
        self.memory_samples = []
        
        # Flatten memory buffer into a list of samples
        for class_id, samples in memory_buffer.items():
            self.memory_samples.extend(samples)
            
        print(f"Created replay dataset with {len(self.current_dataset)} current samples and "
              f"{len(self.memory_samples)} memory samples")
    
    def __len__(self):
        return len(self.current_dataset) + len(self.memory_samples)
    
    def __getitem__(self, idx):
        # Return sample from memory buffer
        if idx < len(self.memory_samples):
            return self.memory_samples[idx]
        
        # Return sample from current dataset
        else:
            adjusted_idx = idx - len(self.memory_samples)
            return self.current_dataset[adjusted_idx]

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

    def __getitem__(self, idx):
        utt_id = self.list_IDs[idx]
        filepath = os.path.join(self.base_dir, utt_id)
        X = load_audio(filepath, self.sample_rate)
        augmethod_index = random.choice(range(len(self.augmentation_methods))) if len(
            self.augmentation_methods) > 0 else -1
        if augmethod_index >= 0:
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
        if self.is_dev_aug:
            print("Dev aug is enabled")

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        filepath = os.path.join(self.base_dir, utt_id)
        X, fs = librosa.load(filepath, sr=16000)
        if self.is_dev_aug:
            augmethod_index = random.choice(range(len(self.augmentation_methods))) if len(
                self.augmentation_methods) > 0 else -1
            if augmethod_index >= 0:
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
        # print("Chunking enabled:", self.enable_chunking)
        # print("trim_length:", trim_length)
        # print("padding_type:", self.padding_type)
        self.no_pad = args.get('no_pad', False) if args is not None else False
        if self.no_pad:
            print('No padding')

    def __getitem__(self, idx):
        utt_id = self.list_IDs[idx]
        filepath = os.path.join(self.base_dir, utt_id)
        X, _ = librosa.load(filepath, sr=16000)
        if self.eval_augment is not None:
            # print("eval_augment:", self.eval_augment)
            X = globals()[self.eval_augment](
                X, self.args, self.sample_rate, audio_path=filepath)
        if not self.enable_chunking and not self.no_pad:
            X = pad(X, padding_type=self.padding_type,
                    max_len=self.trim_length, random_start=self.random_start)
        if self.no_pad and len(X) > 160000:  # 10 seconds is the maximum
            X = X[:160000]
        x_inp = Tensor(X)
        return x_inp, utt_id

class ReplayNormalDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        args: Optional[Dict[str, Any]] = None,
        chunking_eval: bool = False,
        memory_size: int = 2500,  # Size of memory buffer per class
        num_tasks: int = 2,      # Total number of tasks
        #protocol_paths: Optional[List[str]] = None,  # List of protocol files for each task
        skip_first_task: bool = True,
        task0_ratio: float = 0.2,  # Ratio of task 0 samples
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.batch_size_per_device = batch_size
        self.data_dir = data_dir
        self.args = args
        self.protocol_paths = args.get('protocol_paths', None)
        
         # Protocol paths for each task
        if self.protocol_paths is None:
            # Generate default protocol paths
            self.protocol_paths = [
                os.path.join(self.data_dir, f'protocol_task{i}.txt') 
                for i in range(num_tasks)
            ]
        else:
            list_protocol_path = os.listdir(self.protocol_paths)
            # sort the list_protocol_path
            list_protocol_path.sort()
            
            
            dir_path = self.protocol_paths
            # convert self.protocol_paths to dict
            self.protocol_paths = {}
            for i in range(num_tasks):
                self.protocol_paths[i] = os.path.join(dir_path, list_protocol_path[i])
            print("Protocol paths:", self.protocol_paths)
           
            
        
            
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
        
        
        ## Replay-based continual learning code
        
        # Initialize datasets
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.memory_buffer = {}  # Dictionary to store memory samples
        
        # Task tracking
        self.current_task = 0 if not skip_first_task else 1  # Start from task 1 if skipping first task
        self.num_tasks = num_tasks
        self.memory_size = memory_size
        self.skip_first_task = skip_first_task

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Set up datasets for all tasks
        if len(self.train_datasets) == 0:
            for task_id in range(self.num_tasks):
                # Use task-specific protocol file
                protocol_path = self.protocol_paths[task_id]
                print(f"Task {task_id}: Using protocol file {protocol_path}")
                
                # Generate task-specific data lists
                d_label_trn, file_train = self.genList(
                    protocol_path=protocol_path, is_train=True, is_eval=False, is_dev=False)
                
                # if self.skip_first_task and task_id == 0:
                #     # use task0_ratio
                #     file_train = file_train[:int(len(file_train) * task0_ratio)]
                #     d_label_trn = d_label_trn[:int(len(file_train) * task0_ratio)]
                    
                    
                
                print(f'Task {task_id}: no. of training trials', len(file_train))
                
                d_label_dev, file_dev = self.genList(
                    protocol_path=protocol_path, is_train=False, is_eval=False, is_dev=True)
                print(f'Task {task_id}: no. of validation trials', len(file_dev))
                
                d_meta, file_eval = self.genList(
                    protocol_path=protocol_path, is_train=False, is_eval=True, is_dev=False)
                print(f'Task {task_id}: no. of evaluation trials', len(file_eval))
                
                # Create datasets
                train_dataset = Dataset_for(self.args, list_IDs=file_train, labels=d_label_trn,
                                           base_dir=self.data_dir+'/', **self.args)
                
                val_dataset = Dataset_for_dev(self.args, list_IDs=file_dev, labels=d_label_dev,
                                           base_dir=self.data_dir+'/', is_train=False, **self.args)
                
                test_dataset = Dataset_for_eval(self.args, list_IDs=file_eval, labels=None,
                                              base_dir=self.data_dir+'/', random_start=self.args.random_start, 
                                              trim_length=self.args.trim_length, 
                                              repeat_pad=True if self.args.padding_type == 'repeat' else False,
                                              enable_chunking=self.chunking_eval)
                
                self.train_datasets.append(train_dataset)
                self.val_datasets.append(val_dataset)
                self.test_datasets.append(test_dataset)
    # If skipping first task, populate memory buffer with samples from task 0
        
        if self.skip_first_task:
            print("Skipping training on task 0, initializing memory buffer with samples from task 0")
            self._initialize_memory_buffer_from_task0(40)
    def next_task(self):
        """Move to the next task and update memory buffer"""
        if self.current_task < self.num_tasks - 1:
            # Update memory buffer with samples from current task
            self._update_memory_buffer()
            
            # Move to next task
            self.current_task += 1
            print(f"Moving to task {self.current_task}")
        else:
            print("Already at the final task")

    def _initialize_memory_buffer_from_task0(self, num_processes=None):
        """Initialize memory buffer with samples from task 0"""
        task0_dataset = self.train_datasets[0]
        num_classes = 2  # Assuming binary classification

        # Determine the number of processes to use
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()

        # Group samples by class in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            group_samples_partial = partial(_group_samples, task0_dataset)
            class_samples = list(tqdm(pool.imap(group_samples_partial, range(num_classes)), total=num_classes, desc="Grouping samples"))

        # Select samples for memory buffer
        for class_id, samples in enumerate(tqdm(class_samples, desc="Classes")):
            if len(samples) > 0:
                num_samples = min(self.memory_size, len(samples))
                random.shuffle(samples)
                self.memory_buffer[class_id] = samples[:num_samples]
                print(f"Stored {num_samples} samples of class {class_id} from task 0 in memory buffer")

                
    def _update_memory_buffer(self):
        """Store samples from current task in memory buffer"""
        current_dataset = self.train_datasets[self.current_task]
        
        # Group samples by class
        class_samples = {0: [], 1: []}  # 0: spoof, 1: bonafide
        
        for i in range(len(current_dataset)):
            x, y = current_dataset[i]
            class_samples[y].append((x, y, i))
        
        # Randomly select samples for memory buffer
        for class_id in class_samples:
            samples = class_samples[class_id]
            if len(samples) > 0:
                # Determine how many samples to store
                num_samples = min(self.memory_size, len(samples))
                selected_indices = random.sample(range(len(samples)), num_samples)
                
                # Store selected samples
                if class_id not in self.memory_buffer:
                    self.memory_buffer[class_id] = []
                
                for idx in selected_indices:
                    x, y, orig_idx = samples[idx]
                    self.memory_buffer[class_id].append((x, y))
                
                print(f"Stored {num_samples} samples of class {class_id} from task {self.current_task}")
    
    def train_dataloader(self) -> DataLoader[Any]:
        """Return a dataloader for the current task, including memory replay"""
        # Skip task 0 if configured
        if self.skip_first_task and self.current_task == 0:
            # Return empty dataloader - training will be skipped
            return DataLoader(
                dataset=EmptyDataset(),
                batch_size=self.batch_size_per_device
            )
        
        # Current task dataset
        current_dataset = self.train_datasets[self.current_task]
        
        # Create a combined dataset with memory buffer if not at first task or if skipping first task
        if self.current_task > 0 and len(self.memory_buffer) > 0:
            combined_dataset = ReplayDataset(current_dataset, self.memory_buffer)
        else:
            combined_dataset = current_dataset
        
        return DataLoader(
            dataset=combined_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return validation dataloader for current task"""
        return DataLoader(
            dataset=self.val_datasets[self.current_task],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Union[DataLoader[Any], List[DataLoader[Any]]]:
        """Return test dataloaders for all tasks seen so far"""
        # Return a list of dataloaders for all tasks up to current task for evaluation
        dataloaders = []
        for task_id in range(self.current_task + 1):
            dataloaders.append(
                DataLoader(
                    dataset=self.test_datasets[task_id],
                    batch_size=self.batch_size_per_device,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=False,
                    collate_fn=self.eval_collator,
                )
            )
        return dataloaders if len(dataloaders) > 1 else dataloaders[0]

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

    def genList(self, protocol_path, is_train=False, is_eval=False, is_dev=False):
        """Generate file lists for a specific split from a protocol file"""
        # bonafide: 1, spoof: 0
        d_meta = {}
        file_list = []
        
        subset = "train" if is_train else "dev" if is_dev else "eval"

        with open(protocol_path, 'r') as f:
            l_meta = f.readlines()
            
        for line in l_meta:
            utt, data_subset, label = line.strip().split()
            
            # Check if this utterance belongs to the current subset
            if data_subset == subset:
                file_list.append(utt)
                d_meta[utt] = 1 if label == 'bonafide' else 0
                
        return d_meta, file_list

if __name__ == "__main__":
    _ = NormalDataModule()
