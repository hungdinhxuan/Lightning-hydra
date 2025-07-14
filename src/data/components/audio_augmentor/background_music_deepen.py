from .base import BaseAugmentor
from .utils import recursive_list_files, librosa_to_pydub
import random
import numpy as np
import librosa

import logging
#logger = logging.get#logger(__name__)


class BackgroundMusicAugmentor(BaseAugmentor):
    # Class-level cache to store music lists for different paths
    _music_cache = {}
    
    def __init__(self, config: dict):
        """
        Background music augmentation method.
        This augmentation is based on described in the paper:
        "DeePen: Penetration Testing for Audio Deepfake Detection"
        
        Add Background Music: Music from Musan [55] or Free
        Music Archive [56] overlaid at 50% volume.
        
        The implementation:
        1. Randomly selects a music file from the provided directory
        2. Adjusts music duration to match the original audio (repeat/trim)
        3. Reduces music volume to 50% (-6 dB)
        4. Overlays the background music onto the original audio

        Config:
        music_path: str, path to the folder containing music files

        """
        super().__init__(config)
        self.music_path = config["music_path"]
        #print(f"Initializing BackgroundMusicAugmentor with music path: {self.music_path}")
        
        # Use cached music list if available, otherwise compute and cache it
        if self.music_path not in self._music_cache:
            #logger.info(f"First time loading music from {self.music_path}, scanning directory...")
            music_list = self.select_music(self.music_path)
            # Convert to numpy array for faster random access with large lists
            self._music_cache[self.music_path] = np.array(music_list) if music_list else []
            
        self.music_list = self._music_cache[self.music_path]
        
    def select_music(self, music_path: str) -> list:
        """
        Select music files from the given path.
        
        Args:
            music_path: Path to directory containing music files
            
        Returns:
            List of music file paths
        """
        music_list = recursive_list_files(music_path)
        return music_list

    def transform(self):
        """
        Transform the original audio by overlaying background music at 50% volume.
        """
        # Fast random selection using numpy for large lists
        if len(self.music_list) == 0:
            #logger.warning("No music files available, skipping augmentation")
            self.augmented_audio = self.data
            return
            
        # Use numpy's random choice for better performance with large arrays
        if isinstance(self.music_list, np.ndarray):
            selected_music = np.random.choice(self.music_list)
        else:
            selected_music = random.choice(self.music_list)
            
        # #logger.debug(f"Selected music file: {selected_music}")
        music_data, music_sr = librosa.load(selected_music, sr=self.sr)
        
        # Get the lengths of both audio signals
        audio_length = len(self.data)
        music_length = len(music_data)
        
        # #logger.debug(f"Audio length: {audio_length} samples, Music length: {music_length} samples")
        
        # Handle music duration - repeat or trim as needed
        if music_length < audio_length:
            # If music is shorter, repeat it to cover the entire audio
            repeats_needed = (audio_length // music_length) + 1
            music_data = np.tile(music_data, repeats_needed)
            # #logger.debug(f"Music repeated {repeats_needed} times to match audio length")
        
        # Trim music to match the exact length of the original audio
        music_data = music_data[:audio_length]
        
        # Reduce music volume to 50% (multiply by 0.5)
        music_data_50_percent = music_data * 0.5
        
        # Overlay the background music at 50% volume onto the original audio
        augmented_data = self.data + music_data_50_percent
        
        # Convert to pydub for saving (as required by base class)
        self.augmented_audio = librosa_to_pydub(augmented_data, sr=self.sr)
    
    @classmethod
    def clear_cache(cls):
        """Clear the music list cache. Useful when files are added/removed."""
        cls._music_cache.clear()
        #logger.info("Music cache cleared")
    
    @classmethod
    def get_cache_info(cls):
        """Get information about cached music lists."""
        info = {}
        for path, music_list in cls._music_cache.items():
            info[path] = len(music_list)
        return info

