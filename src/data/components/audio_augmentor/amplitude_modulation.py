from .base import BaseAugmentor
import numpy as np
import logging
from .utils import librosa_to_pydub
logger = logging.getLogger(__name__)

class AmplitudeModulationAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Amplitude Modulation augmentor class requires these config:
        min_frequency: float, minimum modulation frequency in Hz (e.g., 0.5)
        max_frequency: float, maximum modulation frequency in Hz (e.g., 5.0)
        """
        super().__init__(config)
        self.min_frequency = config["min_frequency"]
        self.max_frequency = config["max_frequency"]
        self.audio_data = None

    def load(self, input_path: str):
        """
        :param input_path: path to the input audio file
        """
        # load with librosa
        super().load(input_path)
        self.audio_data = self.data

    def transform(self):
        """
        Modulate the amplitude of the audio by multiplying with a sine wave
        """
        # Generate random modulation frequency within the specified range
        mod_frequency = np.random.uniform(self.min_frequency, self.max_frequency)
        
        # Create time array
        duration = len(self.audio_data) / self.sr
        time = np.linspace(0, duration, len(self.audio_data))
        
        # Create modulation signal (sine wave)
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_frequency * time)
        
        # Apply amplitude modulation
        self.augmented_audio = self.audio_data * modulation
        
        # transform to pydub audio segment
        self.augmented_audio = librosa_to_pydub(self.augmented_audio, sr=self.sr)
        