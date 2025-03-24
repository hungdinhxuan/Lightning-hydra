from .base import BaseAugmentor
from .utils import librosa_to_pydub
import random
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment

import logging
logger = logging.getLogger(__name__)
# class PitchAugmentor(BaseAugmentor):
#     """
#         Pitch augmentation
#         Config:
#         min_pitch_shift: int, min pitch shift factor
#         max_pitch_shift: int, max pitch shift factor
#     """
#     def __init__(self, config: dict):
#         """
#         This method initialize the `PitchAugmentor` object.
        
#         :param config: dict, configuration dictionary
#         """
        
#         super().__init__(config)
#         self.min_pitch_shift = config["min_pitch_shift"]
#         self.max_pitch_shift = config["max_pitch_shift"]
#         self.pitch_shift = random.randint(self.min_pitch_shift, self.max_pitch_shift)
        
    
#     def transform(self):
#         """
#         Transform the audio by pitch shifting based on `librosa.effects.pitch_shift`
#         The pitch shift factor is randomly selected between min_pitch_shift and max_pitch_shift
#         """
#         data = librosa.effects.pitch_shift(self.data, sr=self.sr, n_steps=self.pitch_shift)
#         # transform to pydub audio segment
#         self.augmented_audio = librosa_to_pydub(data, sr=self.sr)
# class PitchAugmentor(BaseAugmentor):
#     """
#     Pitch augmentation
#     Generates versions of audio pitch-shifted from -5 to +5 semitones
#     """
#     def __init__(self, config: dict):
#         """
#         This method initializes the `PitchAugmentor` object.
        
#         :param config: dict, configuration dictionary
#         """
#         super().__init__(config)
#         # Define the range of semitones (-5 to +5)
#         self.min_pitch_shift = config["min_pitch_shift"]
#         self.max_pitch_shift = config["max_pitch_shift"]
    
#     def transform_all(self):
#         """
#         Generate all pitch-shifted versions from -5 to +5 semitones
        
#         :return: dict of pitch-shifted audio segments keyed by semitone value
#         """
#         results = {}
        
#         # Generate each semitone shift
#         for n_steps in range(self.min_pitch_shift, self.max_pitch_shift + 1):
#             # Apply pitch shifting
#             shifted_data = librosa.effects.pitch_shift(
#                 y=self.data, 
#                 sr=self.sr, 
#                 n_steps=n_steps,
#                 bins_per_octave=12,  # Standard semitones
#                 res_type="soxr_hq"   # High quality resampling
#             )
            
#             # Transform to pydub audio segment
#             shifted_audio = librosa_to_pydub(shifted_data, sr=self.sr)
#             results[n_steps] = shifted_audio
            
#         return results
    
#     def transform(self, n_steps=None):
#         """
#         Transform the audio by pitch shifting
        
#         :param n_steps: Optional specific semitone shift to apply (-5 to +5)
#                        If None, applies a random shift in the range
#         :return: The pitch-shifted audio segment
#         """
#         # If n_steps not specified, use random value in range
#         if n_steps is None:
#             n_steps = random.randint(self.min_pitch_shift, self.max_pitch_shift)
        
#         # Apply pitch shifting
#         data = librosa.effects.pitch_shift(
#             y=self.data, 
#             sr=self.sr, 
#             n_steps=n_steps,
#             bins_per_octave=12,
#             res_type="soxr_hq"
#         )
        
#         # Transform to pydub audio segment
#         self.augmented_audio = librosa_to_pydub(data, sr=self.sr)
class PitchAugmentor(BaseAugmentor):
    """
    Pitch augmentation
    Generates versions of audio pitch-shifted from -5 to +5 semitones
    """
    def __init__(self, config: dict):
        """
        This method initializes the `PitchAugmentor` object.
        
        :param config: dict, configuration dictionary
        """
        super().__init__(config)
        # Define the range of semitones (-5 to +5)
        self.min_pitch_shift = config["min_pitch_shift"]
        self.max_pitch_shift = config["max_pitch_shift"]
    
    def transform_all(self):
        """
        Generate all pitch-shifted versions from -5 to +5 semitones
        
        :return: dict of pitch-shifted audio segments keyed by semitone value
        """
        results = {}
        
        # Generate each semitone shift
        for n_steps in range(self.min_pitch_shift, self.max_pitch_shift + 1):
            # Apply pitch shifting
            shifted_data = self._apply_pitch_shift(n_steps)
            
            # Transform to pydub audio segment
            shifted_audio = librosa_to_pydub(shifted_data, sr=self.sr)
            results[n_steps] = shifted_audio
            
        return results
    
    def transform(self, n_steps=None):
        """
        Transform the audio by pitch shifting
        
        :param n_steps: Optional specific semitone shift to apply (-5 to +5)
                       If None, applies a random shift in the range
        :return: The pitch-shifted audio segment
        """
        # If n_steps not specified, use random value in range
        if n_steps is None:
            n_steps = random.randint(self.min_pitch_shift, self.max_pitch_shift)
        
        # Apply pitch shifting
        data = self._apply_pitch_shift(n_steps)
        
        # Transform to pydub audio segment
        self.augmented_audio = librosa_to_pydub(data, sr=self.sr)
        
    
    def _apply_pitch_shift(self, n_steps):
        """
        Helper method to apply pitch shifting with appropriate FFT window size
        
        :param n_steps: Number of semitones to shift
        :return: Pitch-shifted audio data
        """
        # Calculate an appropriate n_fft value based on the audio length
        # A good rule of thumb is to use a power of 2 that's smaller than the audio length
        audio_length = len(self.data)
        n_fft = 2048  # Default value
        
        # Find the largest power of 2 that's smaller than the audio length
        if audio_length < n_fft:
            n_fft = 2**int(np.log2(audio_length) - 1)
            # Ensure n_fft is at least 512 for reasonable quality
            n_fft = max(512, n_fft)
        
        # Apply pitch shifting with appropriate n_fft
        return librosa.effects.pitch_shift(
            y=self.data, 
            sr=self.sr, 
            n_steps=n_steps,
            bins_per_octave=12,
            res_type="soxr_hq",
            n_fft=n_fft  # Use the calculated n_fft value
        )
    