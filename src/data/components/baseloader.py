import os
from torch.utils.data import Dataset


class Dataset_base(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=[], 
                 augmentation_methods=[], eval_augment=None, num_additional_real=2, num_additional_spoof=2, 
                 trim_length=64000, wav_samp_rate=16000, noise_path=None, rir_path=None, 
                 aug_dir=None, online_aug=False, repeat_pad=True, is_train=True):
        """
        Args:
            list_IDs (string): Path to the .lst file with real audio filenames.
            vocoders (list): list of vocoder names.
            augmentation_methods (list): List of augmentation methods to apply.
        """
        self.args = args
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.bonafide_dir = os.path.join(base_dir, 'bonafide')
        self.vocoded_dir = os.path.join(base_dir, 'vocoded')
        self.algo = algo
        self.vocoders = vocoders
        print("vocoders:", vocoders)
        
        self.augmentation_methods = augmentation_methods
        if len(augmentation_methods) < 1:
            # using default augmentation method RawBoostWrapper12
            # self.augmentation_methods = ["RawBoost12"]
            print("No augmentation method provided")
        self.eval_augment = eval_augment
        self.num_additional_real = num_additional_real
        self.num_additional_spoof = num_additional_spoof
        self.trim_length = trim_length
        self.sample_rate = wav_samp_rate
        
        self.args.noise_path = noise_path
        self.args.rir_path = rir_path
        self.args.aug_dir = aug_dir
        self.args.online_aug = online_aug
        self.repeat_pad = repeat_pad
        self.is_train = is_train

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        # to be implemented in child classes
        pass
        
