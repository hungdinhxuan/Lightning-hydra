from .base import BaseAugmentor
from .utils import librosa_to_pydub
import numpy as np
import random
import librosa
import soundfile as sf
import os

import logging
logger = logging.getLogger(__name__)

class FrequencyOperationAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Frequency Operation augmentation
        
        This augmentation is based on described in the paper:
        "DeePen: Penetration Testing for Audio Deepfake Detection"
        
        This function applies frequency operations:
        + Freq Minus: Subtracts 0.01–0.1 energy at random STFT frequencies between 0 and 4300 Hz.
        + Freq Plus: Adds 0.01–0.1 energy at random STFT frequencies between 0 and 4300 Hz.
        
        Config:
        operation_type: str, either "freq_minus", "freq_plus", or "random" (default: "random")
        num_operations: int, number of frequency bands to modify (default: 5)
        max_freq: int, maximum frequency to modify in Hz (default: 4300)
        min_energy: float, minimum energy change (default: 0.01)
        max_energy: float, maximum energy change (default: 0.1)
        """
        super().__init__(config)
        self.operation_type = config.get('operation_type', 'random')  # "freq_minus", "freq_plus", or "random"
        self.num_operations = config.get('num_operations', 1)
        self.max_freq = config.get('max_freq', 4300)  # Hz
        self.min_energy = config.get('min_energy', 0.01)
        self.max_energy = config.get('max_energy', 0.1)

    def apply_frequency_operations(self, stft_data):
        """
        Apply frequency operations (Freq Plus/Minus) to the STFT data.
        
        Args:
            stft_data: Complex STFT data (frequency x time)
            
        Returns:
            Modified STFT data
        """
        # Work with magnitude and phase separately
        magnitude = np.abs(stft_data)
        phase = np.angle(stft_data)
        
        # Calculate frequency bins corresponding to max_freq
        freq_bins = magnitude.shape[0]
        nyquist_freq = self.sr // 2
        max_freq_bin = min(int((self.max_freq / nyquist_freq) * freq_bins), freq_bins - 1)
        
        #logger.debug(f"Operating on frequency range: 0-{self.max_freq}Hz (bins 0-{max_freq_bin})")
        
        # Apply multiple frequency operations
        for i in range(self.num_operations):
            # Select random frequency bin within the allowed range
            freq_bin = random.randint(0, max_freq_bin)
            
            # Random energy change between min_energy and max_energy
            energy_change = random.uniform(self.min_energy, self.max_energy)
            
            # Determine operation type
            if self.operation_type == "random":
                operation = random.choice(["freq_minus", "freq_plus"])
            else:
                operation = self.operation_type
            
            # Apply the operation to all time frames at this frequency
            if operation == "freq_minus":
                # Subtract energy (reduce magnitude)
                magnitude[freq_bin, :] = magnitude[freq_bin, :] * (1.0 - energy_change)
                #logger.debug(f"Freq Minus: bin {freq_bin}, energy reduction: {energy_change:.3f}")
            elif operation == "freq_plus":
                # Add energy (increase magnitude)
                magnitude[freq_bin, :] = magnitude[freq_bin, :] * (1.0 + energy_change)
                #logger.debug(f"Freq Plus: bin {freq_bin}, energy increase: {energy_change:.3f}")
        
        # Reconstruct complex STFT data
        modified_stft = magnitude * np.exp(1j * phase)
        return modified_stft

    def transform(self):
        """
        Transform the audio by applying frequency operations.
        Uses STFT as specified in the DeePen paper.
        """
        # Compute STFT of the input audio
        stft_data = librosa.stft(self.data, hop_length=512, n_fft=2048, dtype=np.complex64)
        
        # Apply frequency operations (Freq Plus/Minus)
        modified_stft = self.apply_frequency_operations(stft_data)
        
        # Convert back to time domain using inverse STFT
        augmented_data = librosa.istft(modified_stft, hop_length=512, dtype=np.float32)
        
        # Ensure the output has the same length as input
        # if len(augmented_data) > len(self.data):
        #     augmented_data = augmented_data[:len(self.data)]
        # elif len(augmented_data) < len(self.data):
        #     # Pad with zeros if necessary
        #     augmented_data = np.pad(augmented_data, (0, len(self.data) - len(augmented_data)))
        
        # Store as numpy array (convert to pydub only for saving)
        self.augmented_audio = augmented_data
        