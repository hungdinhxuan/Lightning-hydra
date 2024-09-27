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

'''
   Hemlata Tak, Madhu Kamble, Jose Patino, Massimiliano Todisco, Nicholas Evans.
   RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing.
   In Proc. ICASSP 2022, pp:6382--6386.
'''

def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y

def normWav(x,always):
    if always:
        x = x/np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
            x = x/np.amax(abs(x))
    return x

def genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    b = 1
    for i in range(0, nBands):
        fc = randRange(minF,maxF,0)
        bw = randRange(minBW,maxBW,0)
        c = randRange(minCoeff,maxCoeff,1)
          
        if c/2 == int(c/2):
            c = c + 1
        f1 = fc - bw/2
        f2 = fc + bw/2
        if f1 <= 0:
            f1 = 1/1000
        if f2 >= fs/2:
            f2 =  fs/2-1/1000
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs),b)

    G = randRange(minG,maxG,0) 
    _, h = signal.freqz(b, 1, fs=fs)    
    b = pow(10, G/20)*b/np.amax(abs(h))   
    return b


def filterFIR(x,b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N/2):int(y.shape[0]-N/2)]
    return y

# Linear and non-linear convolutive noise
def LnL_convolutive_noise(x,N_f,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,minBiasLinNonLin,maxBiasLinNonLin,fs):
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG-minBiasLinNonLin
            maxG = maxG-maxBiasLinNonLin
        b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
        y = y + filterFIR(np.power(x, (i+1)),  b)     
    y = y - np.mean(y)
    y = normWav(y,0)
    return y


# Impulsive signal dependent noise
def ISD_additive_noise(x, P, g_sd):
    beta = randRange(0, P, 0)
    
    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len*(beta/100))
    p = np.random.permutation(x_len)[:n]
    f_r= np.multiply(((2*np.random.rand(p.shape[0]))-1),((2*np.random.rand(p.shape[0]))-1))
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y,0)
    return y


# Stationary signal independent noise

def SSI_additive_noise(x,SNRmin,SNRmax,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise,1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = noise / np.linalg.norm(noise,2) * np.linalg.norm(x,2) / 10.0**(0.05 * SNR)
    x = x + noise
    return x

#--------------RawBoost data augmentation algorithms---------------------------##
def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			
class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self,args,list_IDs, labels, base_dir,algo):
        '''self.list_IDs	: list of strings (each string: utt key),
            self.labels      : dictionary (key: utt key, value: label integer)'''
               
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo=algo
        self.args=args
        self.cut=64600 # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):            
        utt_id = self.list_IDs[index]
        X,fs = librosa.load(self.base_dir+utt_id+'.flac', sr=16000) 
        Y=process_Rawboost_feature(X,fs,self.args,self.algo)
        X_pad= pad(Y,self.cut)
        x_inp= Tensor(X_pad)
        target = self.labels[utt_id]

        return x_inp, target
                  
class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir):       
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut=64600 # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+utt_id+'.flac', sr=16000)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp,utt_id  

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
    ) -> None:
        """Initialize a `ASVSpoofDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
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
            args = Args()
            args.database_path = self.data_dir
            track = args.track
            prefix_2021 = 'ASVspoof2021.{}'.format(track)

            d_label_trn,file_train = self.genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),is_train=True,is_eval=False)
            d_label_dev,file_dev = self.genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),is_train=False,is_eval=False)
            file_eval = self.genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_{}_cm_protocols/{}.cm.eval.trl.txt'.format(track,prefix_2021)),is_train=False,is_eval=True)
            
            self.data_train = Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_train/'),algo=args.algo)
            self.data_val = Dataset_ASVspoof2019_train(args,list_IDs = file_dev,labels = d_label_dev,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_dev/'),algo=args.algo)
            self.data_test = Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(args.track)))
            

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
    _ = ASVSpoofDataModule()
