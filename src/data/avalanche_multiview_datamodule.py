from typing import Any, Dict, List, Optional, Tuple
import os
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
from src.data.normal_multiview_datamodule import NormalDataModule, Dataset_for, Dataset_for_dev, Dataset_for_eval
import avalanche as avl
from avalanche.benchmarks import NCScenario, ni_benchmark
from avalanche.training.utils.data_loader import DataLoader as AvalancheDataLoader

class AvalancheMultiViewDataModule(NormalDataModule):
    """DataModule for Avalanche experiments with multi-view data."""
    
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(data_dir, batch_size, num_workers, pin_memory, args)
        self.experience_datasets = {
            'train': [],
            'dev': [],
            'eval': []
        }
        self.current_experience = 0
        
        # Import the collate function for multi-view data
        from src.data.components.collate_fn import multi_view_collate_fn
        self.collate_fn = lambda x: multi_view_collate_fn(
            x,
            self.args.get('views', [1, 2, 3, 4]),
            self.args.get('wav_samp_rate', 16000),
            self.args.get('padding_type', 'repeat'),
            self.args.get('random_start', True),
            self.args.get('view_padding_configs', {})
        )
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and create experience streams for each stage."""
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Load datasets for each experience
        for exp_idx in range(self.args['n_experiences']):
            exp_num = exp_idx + 1
            
            # Process each split (train, dev, eval)
            for split in ['train', 'dev', 'eval']:
                # Construct the path to the protocol file
                protocol_path = os.path.join(
                    self.data_dir,
                    self.args['split_dirs'][split],
                    f'experience_{exp_num}',
                    'protocol.txt'
                )
                
                # Load data for this experience and split
                if split == 'train':
                    d_label, file_list = self.genList(
                        protocol_path=protocol_path,
                        is_train=True,
                        is_eval=False,
                        is_dev=False
                    )
                    dataset = Dataset_for(
                        self.args,
                        list_IDs=file_list,
                        labels=d_label,
                        base_dir=os.path.join(self.data_dir, self.args['split_dirs'][split], f'experience_{exp_num}'),
                        **self.args
                    )
                elif split == 'dev':
                    d_label, file_list = self.genList(
                        protocol_path=protocol_path,
                        is_train=False,
                        is_eval=False,
                        is_dev=True
                    )
                    dataset = Dataset_for_dev(
                        self.args,
                        list_IDs=file_list,
                        labels=d_label,
                        base_dir=os.path.join(self.data_dir, self.args['split_dirs'][split], f'experience_{exp_num}'),
                        is_train=False,
                        **self.args
                    )
                else:  # eval
                    d_label, file_list = self.genList(
                        protocol_path=protocol_path,
                        is_train=False,
                        is_eval=True,
                        is_dev=False
                    )
                    dataset = Dataset_for_eval(
                        self.args,
                        list_IDs=file_list,
                        labels=None,
                        base_dir=os.path.join(self.data_dir, self.args['split_dirs'][split], f'experience_{exp_num}'),
                        random_start=self.args.random_start,
                        trim_length=self.args.trim_length,
                        repeat_pad=True if self.args.padding_type == 'repeat' else False,
                        enable_chunking=False
                    )
                
                # Store the dataset
                self.experience_datasets[split].append(dataset)
    
    def genList(self, protocol_path: str, is_train: bool = False, is_eval: bool = False, is_dev: bool = False):
        """Generate list of files and labels from protocol file."""
        d_meta = {}
        file_list = []
        
        if not os.path.exists(protocol_path):
            raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
        
        with open(protocol_path, 'r') as f:
            l_meta = f.readlines()
            
        for line in l_meta:
            utt, subset, label = line.strip().split()
            if (is_train and subset == 'train') or \
               (is_dev and subset == 'dev') or \
               (is_eval and subset == 'eval'):
                file_list.append(utt)
                d_meta[utt] = 1 if label == 'bonafide' else 0
                
        return d_meta, file_list
    
    def train_dataloader(self) -> DataLoader:
        """Create train dataloader for the current experience."""
        return DataLoader(
            dataset=self.experience_datasets['train'][self.current_experience],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            persistent_workers=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader for the current experience."""
        return DataLoader(
            dataset=self.experience_datasets['dev'][self.current_experience],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader for the current experience."""
        return DataLoader(
            dataset=self.experience_datasets['eval'][self.current_experience],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
    
    def get_experience_stream(self, stage: str) -> List[AvalancheDataLoader]:
        """Get the stream of experiences for a given stage.
        
        Args:
            stage: One of 'train', 'dev', or 'eval'
            
        Returns:
            List of AvalancheDataLoader objects for each experience
        """
        return [
            AvalancheDataLoader(
                dataset,
                batch_size=self.batch_size_per_device,
                shuffle=(stage == 'train'),
                collate_fn=self.collate_fn
            )
            for dataset in self.experience_datasets[stage]
        ]
    
    def set_current_experience(self, exp_idx: int):
        """Set the current experience index.
        
        Args:
            exp_idx: Index of the experience to set as current
        """
        if exp_idx < 0 or exp_idx >= self.args['n_experiences']:
            raise ValueError(f"Experience index {exp_idx} out of range [0, {self.args['n_experiences']-1}]")
        self.current_experience = exp_idx 