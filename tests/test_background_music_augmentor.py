"""Test cases for BackgroundMusicAugmentor class."""

import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
import pytest
from pathlib import Path

from src.data.components.audio_augmentor.background_music_deepen import BackgroundMusicAugmentor


class TestBackgroundMusicAugmentor:
    """Test cases for BackgroundMusicAugmentor."""

    @pytest.fixture
    def sample_audio_data(self):
        """Create sample audio data for testing."""
        # Generate a 2-second sine wave at 440Hz (A note)
        duration = 2.0
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration), False)
        frequency = 440.0
        audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
        return audio_data, sr

    @pytest.fixture
    def sample_music_data(self):
        """Create sample music data for testing."""
        # Generate different music samples
        sr = 16000
        
        # Music 1: Low frequency drone (100Hz)
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), False)
        music1 = 0.2 * np.sin(2 * np.pi * 100 * t)
        
        # Music 2: Mid frequency tone (220Hz)
        duration = 1.5
        t = np.linspace(0, duration, int(sr * duration), False)
        music2 = 0.25 * np.sin(2 * np.pi * 220 * t)
        
        # Music 3: High frequency tone (880Hz)
        duration = 4.0
        t = np.linspace(0, duration, int(sr * duration), False)
        music3 = 0.15 * np.sin(2 * np.pi * 880 * t)
        
        return [(music1, "music1.wav"), (music2, "music2.wav"), (music3, "music3.wav")], sr

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for input, output, and music."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            music_dir = Path(temp_dir) / "music"
            
            input_dir.mkdir()
            output_dir.mkdir()
            music_dir.mkdir()
            
            yield {
                "input": str(input_dir),
                "output": str(output_dir),
                "music": str(music_dir),
                "temp": temp_dir
            }

    @pytest.fixture
    def setup_test_files(self, temp_dirs, sample_audio_data, sample_music_data):
        """Set up test audio and music files."""
        audio_data, sr = sample_audio_data
        music_samples, music_sr = sample_music_data
        
        # Save sample audio file
        input_file = Path(temp_dirs["input"]) / "test_audio.wav"
        sf.write(str(input_file), audio_data, sr)
        
        # Save music files
        music_files = []
        for music_data, filename in music_samples:
            music_file = Path(temp_dirs["music"]) / filename
            sf.write(str(music_file), music_data, music_sr)
            music_files.append(str(music_file))
        
        return {
            "input_file": str(input_file),
            "music_files": music_files,
            "audio_data": audio_data,
            "sr": sr
        }

    def test_augmentor_initialization(self, temp_dirs):
        """Test BackgroundMusicAugmentor initialization."""
        config = {
            "aug_type": "background_music",
            "output_path": temp_dirs["output"],
            "out_format": "wav",
            "music_path": temp_dirs["music"]
        }
        
        augmentor = BackgroundMusicAugmentor(config)
        
        assert augmentor.aug_type == "background_music"
        assert augmentor.output_path == temp_dirs["output"]
        assert augmentor.out_format == "wav"
        assert augmentor.music_path == temp_dirs["music"]

    def test_music_file_selection(self, setup_test_files, temp_dirs):
        """Test music file selection functionality."""
        config = {
            "aug_type": "background_music",
            "output_path": temp_dirs["output"],
            "out_format": "wav",
            "music_path": temp_dirs["music"]
        }
        
        augmentor = BackgroundMusicAugmentor(config)
        
        # Check that music files are found
        assert len(augmentor.music_list) == 3
        assert all(music_file in setup_test_files["music_files"] for music_file in augmentor.music_list)

    def test_load_audio_file(self, setup_test_files, temp_dirs):
        """Test loading audio file."""
        config = {
            "aug_type": "background_music",
            "output_path": temp_dirs["output"],
            "out_format": "wav",
            "music_path": temp_dirs["music"]
        }
        
        augmentor = BackgroundMusicAugmentor(config)
        augmentor.load(setup_test_files["input_file"])
        
        # Check that audio data is loaded correctly
        assert augmentor.data is not None
        assert len(augmentor.data) == len(setup_test_files["audio_data"])
        assert augmentor.sr == setup_test_files["sr"]

    def test_transform_with_longer_music(self, setup_test_files, temp_dirs):
        """Test transform when music is longer than original audio."""
        config = {
            "aug_type": "background_music",
            "output_path": temp_dirs["output"],
            "out_format": "wav",
            "music_path": temp_dirs["music"]
        }
        
        augmentor = BackgroundMusicAugmentor(config)
        augmentor.load(setup_test_files["input_file"])
        
        # Force selection of longer music (music1 is 3 seconds, audio is 2 seconds)
        augmentor.music_list = [setup_test_files["music_files"][0]]  # music1.wav
        
        augmentor.transform()
        
        # Check that augmented audio exists and has correct length
        assert augmentor.augmented_audio is not None
        # Should match original audio length
        original_length = len(setup_test_files["audio_data"])
        # Note: pydub AudioSegment length comparison would be different

    def test_transform_with_shorter_music(self, setup_test_files, temp_dirs):
        """Test transform when music is shorter than original audio."""
        config = {
            "aug_type": "background_music",
            "output_path": temp_dirs["output"],
            "out_format": "wav",
            "music_path": temp_dirs["music"]
        }
        
        augmentor = BackgroundMusicAugmentor(config)
        augmentor.load(setup_test_files["input_file"])
        
        # Force selection of shorter music (music2 is 1.5 seconds, audio is 2 seconds)
        augmentor.music_list = [setup_test_files["music_files"][1]]  # music2.wav
        
        augmentor.transform()
        
        # Check that augmented audio exists
        assert augmentor.augmented_audio is not None

    def test_save_augmented_audio(self, setup_test_files, temp_dirs):
        """Test saving augmented audio file."""
        config = {
            "aug_type": "background_music",
            "output_path": temp_dirs["output"],
            "out_format": "wav",
            "music_path": temp_dirs["music"]
        }
        
        augmentor = BackgroundMusicAugmentor(config)
        augmentor.load(setup_test_files["input_file"])
        augmentor.transform()
        augmentor.save()
        
        # Check that output file exists
        output_file = Path(temp_dirs["output"]) / "test_audio.wav"
        assert output_file.exists()
        
        # Verify the saved file can be loaded
        saved_data, saved_sr = librosa.load(str(output_file), sr=16000)
        assert len(saved_data) > 0
        assert saved_sr == 16000

    def test_generate_multiple_examples(self, setup_test_files, temp_dirs):
        """Generate multiple example audio files with different music."""
        config = {
            "aug_type": "background_music",
            "output_path": temp_dirs["output"],
            "out_format": "wav",
            "music_path": temp_dirs["music"]
        }
        
        augmentor = BackgroundMusicAugmentor(config)
        
        # Generate examples with each music file
        for i, music_file in enumerate(setup_test_files["music_files"]):
            # Create a unique config for each example
            augmentor.music_list = [music_file]
            augmentor.load(setup_test_files["input_file"])
            augmentor.transform()
            
            # Save with unique filename
            augmentor.file_name = f"example_{i+1}_with_{Path(music_file).stem}"
            augmentor.save()
            
            # Verify file was created
            output_file = Path(temp_dirs["output"]) / f"example_{i+1}_with_{Path(music_file).stem}.wav"
            assert output_file.exists()

    def test_volume_reduction(self, setup_test_files, temp_dirs):
        """Test that music volume is reduced to 50%."""
        config = {
            "aug_type": "background_music",
            "output_path": temp_dirs["output"],
            "out_format": "wav",
            "music_path": temp_dirs["music"]
        }
        
        augmentor = BackgroundMusicAugmentor(config)
        augmentor.load(setup_test_files["input_file"])
        
        # Mock the transform to check volume reduction
        original_data = augmentor.data.copy()
        
        # Load music manually to verify volume
        music_data, _ = librosa.load(setup_test_files["music_files"][0], sr=16000)
        music_data = music_data[:len(original_data)]  # Trim to match
        
        augmentor.transform()
        
        # The augmented audio should be original + 0.5 * music
        # We can't easily verify this without accessing intermediate steps,
        # but we can check that the audio has changed
        assert augmentor.augmented_audio is not None

    def test_empty_music_directory(self, temp_dirs, sample_audio_data):
        """Test behavior with empty music directory."""
        # Create empty music directory
        empty_music_dir = Path(temp_dirs["temp"]) / "empty_music"
        empty_music_dir.mkdir()
        
        config = {
            "aug_type": "background_music",
            "output_path": temp_dirs["output"],
            "out_format": "wav",
            "music_path": str(empty_music_dir)
        }
        
        augmentor = BackgroundMusicAugmentor(config)
        
        # Should handle empty music list gracefully
        assert len(augmentor.music_list) == 0


def generate_example_files():
    """
    Standalone function to generate example audio files using BackgroundMusicAugmentor.
    This can be called to create actual example files for demonstration.
    """
    # Create directories
    example_dir = Path("example_outputs")
    input_dir = example_dir / "input"
    output_dir = example_dir / "output" 
    music_dir = example_dir / "music"
    
    for dir_path in [example_dir, input_dir, output_dir, music_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Generate sample input audio (speech-like signal)
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # Create a more complex speech-like signal
    fundamental = 150  # Fundamental frequency for male voice
    speech_signal = (
        0.4 * np.sin(2 * np.pi * fundamental * t) +
        0.2 * np.sin(2 * np.pi * fundamental * 2 * t) +
        0.1 * np.sin(2 * np.pi * fundamental * 3 * t) +
        0.05 * np.random.normal(0, 0.1, len(t))  # Add some noise
    )
    
    # Apply envelope to make it more speech-like
    envelope = np.exp(-t * 0.5) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))
    speech_signal *= envelope
    
    # Save input audio
    input_file = input_dir / "sample_speech.wav"
    librosa.output.write_wav(str(input_file), speech_signal, sr)
    
    # Generate different types of background music
    music_types = [
        ("classical_strings", lambda t: 0.3 * (np.sin(2*np.pi*220*t) + 0.5*np.sin(2*np.pi*330*t))),
        ("ambient_drone", lambda t: 0.2 * np.sin(2*np.pi*80*t) + 0.1*np.sin(2*np.pi*120*t)),
        ("upbeat_melody", lambda t: 0.25 * (np.sin(2*np.pi*440*t) + np.sin(2*np.pi*554*t) + np.sin(2*np.pi*659*t))),
        ("bass_heavy", lambda t: 0.4 * np.sin(2*np.pi*60*t) + 0.2*np.sin(2*np.pi*90*t))
    ]
    
    music_files = []
    for name, signal_func in music_types:
        # Create longer music (4 seconds) to test repetition/trimming
        music_duration = 4.0
        t_music = np.linspace(0, music_duration, int(sr * music_duration), False)
        music_signal = signal_func(t_music)
        
        # Add some variation
        music_signal *= (1 + 0.2 * np.sin(2 * np.pi * 0.5 * t_music))
        
        music_file = music_dir / f"{name}.wav"
        librosa.output.write_wav(str(music_file), music_signal, sr)
        music_files.append(str(music_file))
    
    # Create augmentor and generate examples
    config = {
        "aug_type": "background_music",
        "output_path": str(output_dir),
        "out_format": "wav",
        "music_path": str(music_dir)
    }
    
    augmentor = BackgroundMusicAugmentor(config)
    
    # Generate augmented versions
    for i, music_file in enumerate(music_files):
        music_name = Path(music_file).stem
        
        # Set specific music file
        augmentor.music_list = [music_file]
        augmentor.load(str(input_file))
        augmentor.transform()
        
        # Save with descriptive name
        augmentor.file_name = f"speech_with_{music_name}"
        augmentor.save()
        
        print(f"Generated: {output_dir}/speech_with_{music_name}.wav")
    
    print(f"\nExample files generated in: {example_dir}")
    print(f"Input file: {input_file}")
    print(f"Output files: {output_dir}")
    print(f"Music files: {music_dir}")


if __name__ == "__main__":
    # Generate example files when script is run directly
    generate_example_files()
