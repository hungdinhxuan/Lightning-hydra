from .base import BaseAugmentor
from audiomentations import TimeStretch
import logging
from .utils import librosa_to_pydub
logger = logging.getLogger(__name__)
import random
import librosa
import numpy as np

# class TimeStretchAugmentor(BaseAugmentor):
#     def __init__(self, config: dict):
#         """
#         Time stretch augmentor class requires these config:
#         min_factor: float, min factor
#         max_factor: float, max factor

#         """
#         super().__init__(config)
#         self.my_transform = TimeStretch(
#             min_rate=config["min_factor"],
#             max_rate=config["max_factor"],
#             leave_length_unchanged=True,
#             p=1.0
#         )
#         self.audio_data = None

#     def load(self, input_path: str):
#         """
#         :param input_path: path to the input audio file
#         """
#         # load with librosa
#         super().load(input_path)
#         self.audio_data = self.data

#     def transform(self):
#         """
#         Time stretch the audio using librosa time stretch method
#         """
#         self.augmented_audio = self.my_transform(
#             samples=self.audio_data, sample_rate=self.sr)
class TimeStretchAugmentor(BaseAugmentor):
    """
    Time stretch augmentation
    Modifies audio speed by a factor between 0.8 (slower) and 1.2 (faster)
    using librosa.effects.time_stretch()
    """
    def __init__(self, config: dict = None):
        """
        Initialize the TimeStretchAugmentor
        
        :param config: dict, optional configuration dictionary
                      If None, default range of 0.8 to 1.2 will be used
        """
        super().__init__(config or {})
        
        # Set default values based on the description
        self.min_factor = 0.8
        self.max_factor = 1.2
        
        # Override with config values if provided
        if config:
            self.min_factor = config.get("min_factor", self.min_factor)
            self.max_factor = config.get("max_factor", self.max_factor)
    
    def transform(self, rate=None):
        """
        Time stretch the audio using librosa.effects.time_stretch()
        
        :param rate: Optional specific stretch rate to apply
                    If None, applies a random rate in the specified range
        :return: The time-stretched audio segment
        """
        # If rate not specified, use random value in range
        if rate is None:
            rate = random.uniform(self.min_factor, self.max_factor)
        
        # Apply time stretching using librosa
        stretched_data = librosa.effects.time_stretch(
            y=self.data,
            rate=rate
        )
        
        # Since time_stretch changes audio length, we may need to adjust it
        # to match the original length for some applications
        if len(stretched_data) != len(self.data):
            if len(stretched_data) > len(self.data):
                # If stretched audio is longer, truncate it
                stretched_data = stretched_data[:len(self.data)]
            else:
                # If stretched audio is shorter, pad it (with zeros or repeat)
                padding = np.zeros(len(self.data) - len(stretched_data))
                stretched_data = np.concatenate((stretched_data, padding))
        
        # Transform to pydub audio segment
        self.augmented_audio = stretched_data
    
    def transform_all(self, num_steps=5):
        """
        Generate multiple time-stretched versions across the range
        
        :param num_steps: Number of steps to generate between min and max factors
        :return: dict of time-stretched audio segments keyed by rate value
        """
        results = {}
        
        # Generate evenly spaced rates between min and max factors
        rates = np.linspace(self.min_factor, self.max_factor, num_steps)
        
        # Apply each rate
        for rate in rates:
            # Apply time stretching
            stretched_data = librosa.effects.time_stretch(
                y=self.data,
                rate=rate
            )
            
            # Adjust length if needed (same as in transform method)
            if len(stretched_data) != len(self.data):
                if len(stretched_data) > len(self.data):
                    stretched_data = stretched_data[:len(self.data)]
                else:
                    padding = np.zeros(len(self.data) - len(stretched_data))
                    stretched_data = np.concatenate((stretched_data, padding))
            
            # Transform to pydub audio segment
            stretched_audio = librosa_to_pydub(stretched_data, sr=self.sr)
            results[float(rate)] = stretched_audio
            
        return results
