import torch
from torch.utils.data import Dataset, DataLoader
from avalanche.benchmarks import NCScenario, dataset_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from typing import List, Optional, Tuple, Dict, Any
import os
from src.data.protocol_manager import ProtocolManager, ProtocolEntry

class TaskAwareAudioDataset(Dataset):
    def __init__(
        self,
        entries: List[ProtocolEntry],
        base_dir: str,
        sample_rate: int = 16000,
        **kwargs
    ):
        self.entries = entries
        self.base_dir = base_dir
        self.sample_rate = sample_rate
        self.kwargs = kwargs
        
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        filepath = os.path.join(self.base_dir, entry.path)
        
        # Load audio using your existing load_audio function
        from src.data.components.dataio import load_audio
        X = load_audio(filepath, self.sample_rate)
        
        x_inp = torch.Tensor(X)
        # Convert label to integer (bonafide: 1, spoof: 0)
        target = 1 if entry.label == 'bonafide' else 0
        return x_inp, target, entry.task_id

class AvalancheDataModule:
    def __init__(
        self,
        data_dir: str,
        protocol_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs
    ):
        self.data_dir = data_dir
        self.protocol_manager = ProtocolManager(protocol_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
        
    def prepare_data(self):
        """Prepare the data for continual learning scenarios."""
        # Get all tasks data
        train_data = self.protocol_manager.get_all_tasks_data(subset="train")
        dev_data = self.protocol_manager.get_all_tasks_data(subset="dev")
        test_data = self.protocol_manager.get_all_tasks_data(subset="eval")
        
        # Create datasets for each task
        train_datasets = []
        dev_datasets = []
        test_datasets = []
        
        for task_id in sorted(train_data.keys()):
            # Create train dataset
            if task_id in train_data:
                train_dataset = TaskAwareAudioDataset(
                    train_data[task_id],
                    self.data_dir,
                    **self.kwargs
                )
                train_datasets.append(AvalancheDataset(train_dataset))
            
            # Create dev dataset
            if task_id in dev_data:
                dev_dataset = TaskAwareAudioDataset(
                    dev_data[task_id],
                    self.data_dir,
                    **self.kwargs
                )
                dev_datasets.append(AvalancheDataset(dev_dataset))
            
            # Create test dataset
            if task_id in test_data:
                test_dataset = TaskAwareAudioDataset(
                    test_data[task_id],
                    self.data_dir,
                    **self.kwargs
                )
                test_datasets.append(AvalancheDataset(test_dataset))
        
        # Create benchmark
        self.benchmark = dataset_benchmark(
            train_datasets,
            dev_datasets,
            test_datasets
        )
        
    def get_benchmark(self):
        """Get the Avalanche benchmark."""
        return self.benchmark
    
    def get_task_info(self):
        """Get information about tasks."""
        return {
            'num_tasks': len(self.protocol_manager.task_protocols),
            'task_labels': self.protocol_manager.get_all_task_labels()
        } 