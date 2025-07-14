from .base import BaseAugmentor
import numpy as np
import logging
from .utils import librosa_to_pydub
logger = logging.getLogger(__name__)


class EchoAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Echo augmentor class requires these config:
        min_delay: float, minimum delay in seconds (e.g., 0.1)
        max_delay: float, maximum delay in seconds (e.g., 1.0)
        min_decay: float, minimum decay factor (e.g., 0.3)
        max_decay: float, maximum decay factor (e.g., 0.9)
        """
        super().__init__(config)
        # Since audiomentations doesn't have a direct Echo transform,
        # we'll implement it in the transform method
        self.min_delay = config["min_delay"]
        self.max_delay = config["max_delay"]
        self.min_decay = config["min_decay"]
        self.max_decay = config["max_decay"]
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
        Add echo to the audio with random delay and decay values
        """
        # Generate random delay and decay values within the specified ranges
        delay_samples = int(np.random.uniform(self.min_delay, self.max_delay) * self.sr)
        decay = np.random.uniform(self.min_decay, self.max_decay)
        
        # Create echo effect
        echo_audio = np.copy(self.audio_data)
        if delay_samples < len(self.audio_data):
            echo_audio[delay_samples:] += decay * self.audio_data[:-delay_samples]
        
        self.augmented_audio = echo_audio
        self.augmented_audio = librosa_to_pydub(self.augmented_audio, sr=self.sr)
        