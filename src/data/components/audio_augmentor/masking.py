from .base import BaseAugmentor
from .utils import librosa_to_pydub
import numpy as np
import random
import librosa
import soundfile as sf
import os

import logging
logger = logging.getLogger(__name__)

class MaskingAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Masking augmentation
        Config:
        F: int, maximum frequency mask parameter
        num_frequency_masks: int, number of frequency masks to apply
        T: int, maximum time mask parameter (number of frames to mask)
        p: float, upper bound parameter for time mask as a fraction of total time steps
        num_time_masks: int, number of time masks to apply
        """
        super().__init__(config)
        self.F = config.get('F', 27)
        self.num_frequency_masks = config.get('num_frequency_masks', 2)
        self.T = config.get('T', 100)
        self.p = config.get('p', 1.0)
        self.num_time_masks = config.get('num_time_masks', 2)

    def apply_frequency_masking(self, mel_spectrogram):
        """
        Apply frequency masking to the mel spectrogram.
        """
        num_mel_channels = mel_spectrogram.shape[0]
        
        for _ in range(self.num_frequency_masks):
            f = random.randint(0, self.F)
            f0 = random.randint(0, num_mel_channels - f)
            mel_spectrogram[f0:f0 + f, :] = 0
        
        return mel_spectrogram

    def apply_time_masking(self, mel_spectrogram):
        """
        Apply time masking to the mel spectrogram.
        """
        num_time_steps = mel_spectrogram.shape[1]
        max_mask_length = int(self.p * num_time_steps)

        for _ in range(self.num_time_masks):
            if num_time_steps > 1:
                t = random.randint(1, min(self.T, max_mask_length))
                t0 = random.randint(0, num_time_steps - t)
                mel_spectrogram[:, t0:t0 + t] = 0

        return mel_spectrogram

    def transform(self):
        """
        Transform the audio by applying frequency and time masking.
        """
        data = self.data.copy()
        mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=self.sr)

        masked_mel_spectrogram = self.apply_frequency_masking(mel_spectrogram)
        masked_mel_spectrogram = self.apply_time_masking(masked_mel_spectrogram)
        
        masked_audio = librosa.feature.inverse.mel_to_audio(masked_mel_spectrogram, sr=self.sr)
        self.augmented_audio = librosa_to_pydub(masked_audio, sr=self.sr)