from .base import BaseAugmentor
from .utils import recursive_list_files, librosa_to_pydub
from audiomentations import AddBackgroundNoise, PolarityInversion
import random
import numpy as np
import librosa

import logging
#logger = logging.get#logger(__name__)


class BackgroundNoiseAugmentor(BaseAugmentor):
    # Class-level cache to store noise lists for different paths
    _noise_cache = {}
    
    def __init__(self, config: dict):
        """
        Background noise augmentation method.
        This augmentation is based on described in the paper:
        "DeePen: Penetration Testing for Audio Deepfake Detection"
        
        Add Background Noise: Noise from Noise ESC 50 [57] or
        Musan [55] datasets added at 50% relative volume.
        
        The implementation:
        1. Randomly selects a noise file from the provided directory
        2. Adjusts noise duration to match the original audio (repeat/trim)
        3. Reduces noise volume to 50%
        4. Overlays the background noise onto the original audio

        Config:
        noise_path: str, path to the folder containing noise files
        
        """
        super().__init__(config)
        self.noise_path = config["noise_path"]
        self.noise_augmentor = AddBackgroundNoise(
            sounds_path=self.noise_path,
            min_snr_db=config["min_snr_db"] if "min_snr_db" in config else 3,
            max_snr_db=config["max_snr_db"] if "max_snr_db" in config else 30,
            noise_transform=PolarityInversion(),
            p=1.0
        )
        
        
    def transform(self):
        """
        Transform the original audio by overlaying background noise at 50% volume.
        """
        augmented_data = self.noise_augmentor(self.data, sample_rate=self.sr)
        self.augmented_audio = librosa_to_pydub(augmented_data, sr=self.sr)
    
    
