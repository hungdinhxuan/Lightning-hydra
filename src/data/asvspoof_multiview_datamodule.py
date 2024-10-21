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
from src.data.components.RawBoost import process_Rawboost_feature
from src.data.components.dataio import load_audio, pad
from src.data.components.collate_fn import multi_view_collate_fn
'''
   Hemlata Tak, Madhu Kamble, Jose Patino, Massimiliano Todisco, Nicholas Evans.
   RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing.
   In Proc. ICASSP 2022, pp:6382--6386.
'''

class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self,args,list_IDs, labels, base_dir,algo):
        '''self.list_IDs	: list of strings (each string: utt key),
            self.labels      : dictionary (key: utt key, value: label integer)'''
               
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo=algo
        self.args=args
        self.fs = args.get('sampling_rate', 16000) if args is not None else 16000        

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):            
        utt_id = self.list_IDs[index]
        X,fs = librosa.load(self.base_dir+utt_id+'.flac', sr=self.fs) 
        Y=process_Rawboost_feature(X,fs,self.args,self.algo)
        x_inp= Tensor(Y)
        target = self.labels[utt_id]
        return x_inp, target
                  
class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, args, list_IDs, base_dir):       
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        
        # Sampling rate and cut-off
        self.fs = args.get('sampling_rate', 16000) if args is not None else 16000
        self.cut = args.get('cut', 64600) if args is not None else 64600
        self.padding_type = args.get('padding_type', 'zero') if args is not None else 'zero'
        self.random_start = args.get('random_start', False) if args is not None else False
        print('padding_type:',self.padding_type)
        print('cut:',self.cut)
        print('random_start:',self.random_start)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+utt_id+'.flac', sr=self.fs)
        X_pad = pad(X,self.padding_type, self.cut, self.random_start)
        x_inp = Tensor(X_pad)
        return x_inp,utt_id  


class ASVSpoofDataModule(LightningDataModule):
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
        self.args = args
        self.protocols_path = self.args.get('protocols_path', '/data/hungdx/Datasets/protocols/database/') if self.args is not None else '/data/hungdx/Datasets/protocols/database/'

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
            
            self.database_path = self.data_dir
            track = 'DF'
            
            prefix_2021 = 'ASVspoof2021.{}'.format(track)
            self.algo = self.args.get('algo', -1) if self.args is not None else -1

            d_label_trn,file_train = self.genSpoof_list( dir_meta =  os.path.join(self.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),is_train=True,is_eval=False)
            d_label_dev,file_dev = self.genSpoof_list( dir_meta =  os.path.join(self.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),is_train=False,is_eval=False)
            file_eval = self.genSpoof_list( dir_meta =  os.path.join(self.protocols_path+'ASVspoof_{}_cm_protocols/{}.cm.eval.trl.txt'.format(track,prefix_2021)),is_train=False,is_eval=True)
            
            self.data_train = Dataset_ASVspoof2019_train(self.args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(self.database_path+'ASVspoof2019_LA_train/'),algo=self.algo)
            self.data_val = Dataset_ASVspoof2019_train(self.args,list_IDs = file_dev,labels = d_label_dev,base_dir = os.path.join(self.database_path+'ASVspoof2019_LA_dev/'),algo=self.algo)
            self.data_test = Dataset_ASVspoof2021_eval(self.args, list_IDs = file_eval,base_dir = os.path.join(self.database_path+'ASVspoof2021_{}_eval/'.format(track)))
            

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
            collate_fn=lambda x: multi_view_collate_fn(x, self.args.views, self.args.sample_rate, self.args.padding_type, self.args.random_start),
            drop_last=True,
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
            collate_fn=lambda x: multi_view_collate_fn(x, self.args.views, self.args.sample_rate, self.args.padding_type, self.args.random_start),
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
    _ = ASVSpoofDataModule()
