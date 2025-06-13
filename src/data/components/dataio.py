import os
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from typing import Union
import shutil


def get_cache_path(file_path: str, cache_dir: str) -> str:
    """
    Convert file path to cache path
    Example:
        input: /nvme1/0.wav
        output: cache_dir/nvme1/0.npy
    """
    # Get the directory and filename without extension
    dir_name = os.path.dirname(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Create cache directory structure
    cache_subdir = os.path.join(cache_dir, dir_name.lstrip('/'))
    os.makedirs(cache_subdir, exist_ok=True)
    
    # Return full cache path
    return os.path.join(cache_subdir, f"{file_name}.npy")


def load_audio(file_path: str, sr: int = 16000, cache_dir: str = None) -> np.ndarray:
    '''
    Load audio file with caching support
    file_path: path to the audio file
    sr: sampling rate, default 16000
    cache_dir: directory to store cached audio files, if None caching is disabled
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    # If caching is enabled, try to load from cache first
    if cache_dir is not None:
        cache_path = get_cache_path(file_path, cache_dir)
        
        # Try to load from cache
        if os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception as e:
                print(f"Error loading cache file {cache_path}: {e}")
                # If cache loading fails, continue with normal loading
        
        # Load audio and cache it
        audio, _ = librosa.load(file_path, sr=sr)
        try:
            np.save(cache_path, audio)
        except Exception as e:
            print(f"Error saving cache file {cache_path}: {e}")
        return audio
    
    # If caching is disabled, load normally
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


def load_torchaudio(file_path: str, sr: int = 16000) -> torch.Tensor:
    '''
    Load audio file
    file_path: path to the audio file
    sr: sampling rate, default 16000
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    audio, sample_rate = torchaudio.load(file_path)
    if sample_rate != sr:
        raise ValueError(f"Sample rate mismatch: {sample_rate} != {sr}")
    return audio


def save_audio(file_path: str, audio: np.ndarray, sr: int = 16000):
    '''
    Save audio file
    file_path: path to save the audio file
    audio: audio signal
    sr: sampling rate, default 16000
    '''
    sf.write(file_path, audio, sr, subtype='PCM_16')


def npwav2torch(waveform: np.ndarray) -> torch.Tensor:
    '''
    Convert numpy array to torch tensor
    waveform: audio signal
    '''
    return torch.from_numpy(waveform).float()


def pad(x: np.ndarray, padding_type: str = 'zero', max_len=64000, random_start=False) -> np.ndarray:
    '''
    pad audio signal to max_len
    x: audio signal
    padding_type: 'zero' or 'repeat' when len(X) < max_len, default 'zero'
        zero: pad with zeros
        repeat: repeat the signal until it reaches max_len
    max_len: max length of the audio, default 64000
    random_start: if True, randomly choose the start point of the audio
    '''
    # Ensure that max_len should be integer
    max_len = int(max_len)
    x_len = x.shape[0]
    padded_x = None
    if max_len == 0:
        # no padding
        print("Warning: max_len is 0, no padding will be applied")
        padded_x = x
    elif max_len > 0:
        if x_len >= max_len:
            if random_start:
                start = np.random.randint(0, x_len - max_len+1)
                padded_x = x[start:start + max_len]
            else:
                padded_x = x[:max_len]
        else:
            if random_start:
                # keep at least half of the signal
                start = np.random.randint(0, int((x_len+1)/2))
                x_new = x[start:]
            else:
                x_new = x

            if padding_type == "repeat":
                num_repeats = int(max_len / len(x_new)) + 1
                padded_x = np.tile(x_new, (1, num_repeats))[:, :max_len][0]

            elif padding_type == "zero":
                padded_x = np.zeros(max_len)
                padded_x[:len(x_new)] = x_new

    else:
        raise ValueError("max_len must be >= 0")

    return padded_x


def pad_tensor(x: torch.Tensor, padding_type: str = 'zero', max_len: int = 64000, random_start: bool = False) -> torch.Tensor:
    '''
    Pad audio signal to max_len.

    Args:
        x: audio signal
        padding_type: 'zero' or 'repeat' when len(X) < max_len, default 'zero'
            zero: pad with zeros
            repeat: repeat the signal until it reaches max_len
        max_len: max length of the audio, default 64000
        random_start: if True, randomly choose the start point of the audio

    Returns:
        padded_x: Padded audio signal
    '''
    x_len = x.shape[0]
    padded_x = None

    if max_len == 0:
        # no padding
        padded_x = x
    elif max_len > 0:
        if x_len >= max_len:
            if random_start:
                start = torch.randint(0, x_len - max_len + 1, (1,)).item()
                padded_x = x[start:start + max_len]
            else:
                padded_x = x[:max_len]
        else:
            if random_start:
                start = torch.randint(0, max_len - x_len + 1, (1,)).item()
                if padding_type == "repeat":
                    num_repeats = (max_len // x_len) + 1
                    padded_x = x.repeat(num_repeats)[start:start + max_len]
                elif padding_type == "zero":
                    padded_x = torch.zeros(max_len, dtype=x.dtype)
                    padded_x[start:start + x_len] = x
            else:
                if padding_type == "repeat":
                    num_repeats = (max_len // x_len) + 1
                    padded_x = x.repeat(num_repeats)[:max_len]
                elif padding_type == "zero":
                    padded_x = torch.zeros(max_len, dtype=x.dtype)
                    padded_x[:x_len] = x
    else:
        raise ValueError("max_len must be >= 0")

    return padded_x
