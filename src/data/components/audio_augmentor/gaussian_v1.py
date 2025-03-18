from .base import BaseAugmentor
from .utils import librosa_to_pydub
import random
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import wave
import logging

logger = logging.getLogger(__name__)

class GaussianAugmentor(BaseAugmentor):
    """
    Gaussian noise augmentation
    
    Adds Gaussian noise with mean 0 and a standard
    deviation randomly selected between 0.01 and 0.2 to the audio.
    
    Config:
    min_std_dev: float, minimum standard deviation (default: 0.01)
    max_std_dev: float, maximum standard deviation (default: 0.2)
    """
    def __init__(self, config: dict):
        """
        This method initializes the `GaussianAugmentor` object.
        
        :param config: dict, configuration dictionary
        """
        super().__init__(config)
        self.min_std_dev = config.get('min_std_dev', 0.01)
        self.max_std_dev = config.get('max_std_dev', 0.2)
        self.mean =config.get('mean', 0)
        assert self.min_std_dev > 0.0
        assert self.max_std_dev > 0.0
        assert self.max_std_dev >= self.min_std_dev
        self.std_dev = random.uniform(self.min_std_dev, self.max_std_dev)
        
    def transform(self):
        """
        Transform the audio by adding Gaussian noise with mean 0
        and randomly selected standard deviation.
        """
        # Generate Gaussian noise with mean and selected std_dev
        noise = np.random.normal(self.mean, self.std_dev, self.data.shape[0]).astype(np.float32)
        
        self.augmented_audio = self.data + noise
