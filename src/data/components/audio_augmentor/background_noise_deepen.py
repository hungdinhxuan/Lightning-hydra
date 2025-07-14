from .base import BaseAugmentor
from .utils import recursive_list_files, librosa_to_pydub
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
        #print(f"Initializing BackgroundNoiseAugmentor with noise path: {self.noise_path}")
        
        # Use cached noise list if available, otherwise compute and cache it
        if self.noise_path not in self._noise_cache:
            #logger.info(f"First time loading noise from {self.noise_path}, scanning directory...")
            noise_list = self.select_noise(self.noise_path)
            # Convert to numpy array for faster random access with large lists
            self._noise_cache[self.noise_path] = np.array(noise_list) if noise_list else []

            
        self.noise_list = self._noise_cache[self.noise_path]
        

    def select_noise(self, noise_path: str) -> list:
        """
        Select noise files from the given path.
        
        Args:
            noise_path: Path to directory containing noise files
            
        Returns:
            List of noise file paths
        """
        noise_list = recursive_list_files(noise_path)

        return noise_list

    def transform(self):
        """
        Transform the original audio by overlaying background noise at 50% volume.
        """
 
            
        # Fast random selection using numpy for large lists
        if len(self.noise_list) == 0:
            ##logger.warning("No noise files available, skipping augmentation")
            self.augmented_audio = self.data
            return
            
        # Use numpy's random choice for better performance with large arrays
        if isinstance(self.noise_list, np.ndarray):
            selected_noise = np.random.choice(self.noise_list)
        else:
            selected_noise = random.choice(self.noise_list)
        
        noise_data, noise_sr = librosa.load(selected_noise, sr=self.sr)
        
        # Get the lengths of both audio signals
        audio_length = len(self.data)
        noise_length = len(noise_data)
        
        # Handle noise duration - repeat or trim as needed
        if noise_length < audio_length:
            # If noise is shorter, repeat it to cover the entire audio
            repeats_needed = (audio_length // noise_length) + 1
            noise_data = np.tile(noise_data, repeats_needed)
            
        
        # Trim noise to match the exact length of the original audio
        noise_data = noise_data[:audio_length]
        
        # Reduce noise volume to 50% (multiply by 0.5)
        noise_data_50_percent = noise_data * 0.5
        
        # Overlay the background noise at 50% volume onto the original audio
        augmented_data = self.data + noise_data_50_percent

        self.augmented_audio = librosa_to_pydub(augmented_data, sr=self.sr)
    
    @classmethod
    def clear_cache(cls):
        """Clear the noise list cache. Useful when files are added/removed."""
        cls._noise_cache.clear()
        #logger.info("Noise cache cleared")
    
    @classmethod
    def get_cache_info(cls):
        """Get information about cached noise lists."""
        info = {}
        for path, noise_list in cls._noise_cache.items():
            info[path] = len(noise_list)
        return info

