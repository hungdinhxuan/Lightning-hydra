#!/usr/bin/env python3
"""
Simple example script to test GaussianAugmentor with different noise levels.
Generates augmented audio files for manual quality checking.
"""

import os
import numpy as np
import librosa
from src.data.components.audio_augmentor.gaussian_v1 import GaussianAugmentor
from src.data.components.audio_augmentor.utils import librosa_to_pydub

def create_test_audio(duration=3.0, sample_rate=16000, frequency=440):
    """Create a simple sine wave test audio."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a simple sine wave (A4 note)
    audio = np.sin(frequency * 2 * np.pi * t).astype(np.float32)
    return audio, sample_rate

def test_gaussian_augmentor():
    """Test GaussianAugmentor with different noise levels."""
    
    # Create output directory
    output_dir = "gaussian_test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test configurations with different noise levels
    test_configs = [
        {
            "aug_type": "gaussian",
            "output_path": output_dir,
            "out_format": "wav",
            "min_std_dev": 0.01,
            "max_std_dev": 0.01,
            "mean": 0
        },
        {
            "aug_type": "gaussian", 
            "output_path": output_dir,
            "out_format": "wav",
            "min_std_dev": 0.05,
            "max_std_dev": 0.05,
            "mean": 0
        },
        {
            "aug_type": "gaussian",
            "output_path": output_dir, 
            "out_format": "wav",
            "min_std_dev": 0.1,
            "max_std_dev": 0.1,
            "mean": 0
        },
        {
            "aug_type": "gaussian",
            "output_path": output_dir,
            "out_format": "wav", 
            "min_std_dev": 0.2,
            "max_std_dev": 0.2,
            "mean": 0
        }
    ]
    
    # Create test audio
    print("Creating test audio...")
    test_audio, sr = create_test_audio()
    
    # Save original test audio for comparison
    original_audio_segment = librosa_to_pydub(test_audio, sr)
    original_audio_segment.export(os.path.join(output_dir, "original.wav"), format="wav")
    print(f"Saved original audio: {output_dir}/original.wav")
    
    # Test each configuration
    for i, config in enumerate(test_configs):
        print(f"\nTesting configuration {i+1}: std_dev={config['min_std_dev']}")
        
        # Create augmentor
        augmentor = GaussianAugmentor(config)
        
        # Set the audio data directly (since we're not loading from file)
        augmentor.data = test_audio
        augmentor.sr = sr
        augmentor.file_name = f"gaussian_test_std_{config['min_std_dev']}"
        
        # Apply augmentation
        augmentor.transform()
        
        # Convert to pydub format for saving
        augmentor.augmented_audio = librosa_to_pydub(augmentor.augmented_audio, sr)
        
        # Save augmented audio
        augmentor.save()
        
        output_file = os.path.join(output_dir, f"{augmentor.file_name}.wav")
        print(f"Saved augmented audio: {output_file}")
    
    print(f"\nAll test files generated in: {output_dir}/")
    print("Files generated:")
    print("- original.wav (reference)")
    print("- gaussian_test_std_0.01.wav (very light noise)")
    print("- gaussian_test_std_0.05.wav (light noise)")
    print("- gaussian_test_std_0.1.wav (moderate noise)")
    print("- gaussian_test_std_0.2.wav (heavy noise)")

def test_with_real_audio(audio_file_path):
    """Test GaussianAugmentor with a real audio file."""
    if not os.path.exists(audio_file_path):
        print(f"Audio file not found: {audio_file_path}")
        return
    
    output_dir = "gaussian_real_audio_test"
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        "aug_type": "gaussian",
        "output_path": output_dir,
        "out_format": "wav",
        "min_std_dev": 0.05,
        "max_std_dev": 0.05,
        "mean": 0
    }
    
    print(f"Testing with real audio file: {audio_file_path}")
    
    # Create augmentor and load audio
    augmentor = GaussianAugmentor(config)
    augmentor.load(audio_file_path)
    
    # Apply augmentation
    augmentor.transform()
    
    # Convert to pydub format for saving
    augmentor.augmented_audio = librosa_to_pydub(augmentor.augmented_audio, augmentor.sr)
    
    # Save augmented audio
    augmentor.save()
    
    print(f"Augmented audio saved: {output_dir}/{augmentor.file_name}.wav")

if __name__ == "__main__":
    print("=== Gaussian Augmentor Test ===")
    
    # Test with synthetic audio
    test_gaussian_augmentor()
    
    # Uncomment and provide a real audio file path to test with real audio
    # test_with_real_audio("path/to/your/audio.wav") 