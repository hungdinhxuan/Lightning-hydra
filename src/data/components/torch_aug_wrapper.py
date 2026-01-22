"""
Wrapper module for torch-audiomentations augmentation methods.

This module provides a unified interface to use torch-audiomentations
augmentations with the existing codebase that works with numpy arrays
and single audio files.

All wrapper functions follow the signature:
    def augmentation_name(x: np.ndarray, args: dict, sr: int = 16000, audio_path: str = None) -> np.ndarray

Where:
    - x: numpy array containing audio waveform (1D array)
    - args: dictionary containing configuration and paths
    - sr: sample rate (default: 16000)
    - audio_path: path to audio file (optional, mainly for caching)
    
Returns:
    - numpy array containing augmented audio waveform
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, Callable, Union
import os

# Import torch-audiomentations augmentations
try:
    from torch_audiomentations import (
        AddBackgroundNoise,
        AddColoredNoise,
        ApplyImpulseResponse,
        BandPassFilter,
        BandStopFilter,
        Gain,
        HighPassFilter,
        LowPassFilter,
        PeakNormalization,
        PitchShift,
        PolarityInversion,
        Shift,
        TimeInversion,
        Compose,
        OneOf,
        SomeOf,
    )
    from torch_audiomentations.utils.io import Audio
    TORCH_AUDIO_AVAILABLE = True
except ImportError:
    TORCH_AUDIO_AVAILABLE = False
    print("Warning: torch-audiomentations not available. Install with: pip install torch-audiomentations")


def numpy_to_torch_audio(x: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
    """
    Convert numpy array (1D) to torch tensor format (batch, channels, samples).
    
    Args:
        x: numpy array of shape (samples,)
        sample_rate: sample rate of the audio
        
    Returns:
        torch.Tensor of shape (1, 1, samples)
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x.copy()).float()
    
    # Ensure it's 1D
    if x.ndim == 0:
        x = x.unsqueeze(0)
    elif x.ndim > 1:
        x = x.flatten()
    
    # Add batch and channel dimensions: (samples,) -> (1, 1, samples)
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    
    return x


def torch_audio_to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor (batch, channels, samples) to numpy array (1D).
    
    Args:
        x: torch.Tensor of shape (batch, channels, samples) or (1, 1, samples)
        
    Returns:
        numpy array of shape (samples,)
    """
    # Convert to numpy if tensor
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    
    # Remove batch and channel dimensions: (1, 1, samples) -> (samples,)
    while x.ndim > 1:
        x = x.squeeze(0)
    
    # Ensure it's 1D
    if x.ndim == 0:
        x = np.array([x])
    
    return x.flatten()


def create_torch_aug_wrapper(
    aug_class: type,
    config_func: Optional[Callable[[Dict[str, Any], int, Optional[str]], Dict[str, Any]]] = None,
    **default_kwargs
) -> Callable:
    """
    Create a wrapper function for a torch-audiomentations augmentation.
    
    This factory function creates a wrapper that:
    1. Converts numpy array to torch tensor format
    2. Initializes the augmentation with configuration
    3. Applies the augmentation
    4. Converts result back to numpy array
    
    Args:
        aug_class: The torch-audiomentations augmentation class
        config_func: Optional function to create config from args
                    Signature: config_func(args, sr, audio_path) -> dict
        **default_kwargs: Default keyword arguments for the augmentation
        
    Returns:
        Wrapper function with signature: (x, args, sr, audio_path) -> np.ndarray
    """
    if not TORCH_AUDIO_AVAILABLE:
        def unavailable_wrapper(x, args, sr=16000, audio_path=None):
            raise ImportError("torch-audiomentations is not available")
        return unavailable_wrapper
    
    def wrapper(x: np.ndarray, args: Dict[str, Any], sr: int = 16000, audio_path: Optional[str] = None) -> np.ndarray:
        """
        Wrapper function for torch-audiomentations augmentation.
        
        Args:
            x: numpy array containing audio waveform
            args: dictionary containing configuration
            sr: sample rate
            audio_path: path to audio file (for caching if needed)
            
        Returns:
            numpy array containing augmented audio waveform
        """
        # Create configuration
        if config_func:
            aug_config = config_func(args, sr, audio_path)
        else:
            aug_config = {}
        
        # Merge with default kwargs
        aug_config = {**default_kwargs, **aug_config}
        
        # Add required parameters
        if 'sample_rate' not in aug_config:
            aug_config['sample_rate'] = sr
        if 'output_type' not in aug_config:
            aug_config['output_type'] = 'dict'
        
        # Create augmentation instance
        augment = aug_class(**aug_config)
        
        # Convert numpy to torch format
        x_torch = numpy_to_torch_audio(x, sr)
        
        # Apply augmentation
        result = augment(samples=x_torch, sample_rate=sr)
        
        # Handle different output types
        if isinstance(result, dict):
            result_tensor = result['samples']
        elif isinstance(result, torch.Tensor):
            result_tensor = result
        else:
            result_tensor = result.samples
        
        # Convert back to numpy
        return torch_audio_to_numpy(result_tensor)
    
    return wrapper


# ============================================================================
# Individual augmentation wrappers
# ============================================================================

def gain(x: np.ndarray, args: Dict[str, Any], sr: int = 16000, audio_path: Optional[str] = None) -> np.ndarray:
    """Gain augmentation wrapper."""
    def config_func(args, sr, audio_path):
        return {
            'min_gain_in_db': args.get('min_gain_in_db', -6.0),
            'max_gain_in_db': args.get('max_gain_in_db', 6.0),
            'p': args.get('p', 1.0),
            'mode': args.get('mode', 'per_example'),
        }
    return create_torch_aug_wrapper(Gain, config_func)(x, args, sr, audio_path)


def polarity_inversion(x: np.ndarray, args: Dict[str, Any], sr: int = 16000, audio_path: Optional[str] = None) -> np.ndarray:
    """Polarity inversion augmentation wrapper."""
    def config_func(args, sr, audio_path):
        return {'p': args.get('p', 1.0)}
    return create_torch_aug_wrapper(PolarityInversion, config_func)(x, args, sr, audio_path)


def peak_normalization(x: np.ndarray, args: Dict[str, Any], sr: int = 16000, audio_path: Optional[str] = None) -> np.ndarray:
    """Peak normalization augmentation wrapper."""
    def config_func(args, sr, audio_path):
        return {
            'p': args.get('p', 1.0),
            'apply_to': args.get('apply_to', 'all'),
        }
    return create_torch_aug_wrapper(PeakNormalization, config_func)(x, args, sr, audio_path)


def shift(x: np.ndarray, args: Dict[str, Any], sr: int = 16000, audio_path: Optional[str] = None) -> np.ndarray:
    """Shift augmentation wrapper."""
    def config_func(args, sr, audio_path):
        return {
            'min_shift': args.get('min_shift', -0.5),
            'max_shift': args.get('max_shift', 0.5),
            'shift_unit': args.get('shift_unit', 'fraction'),
            'rollover': args.get('rollover', True),
            'p': args.get('p', 1.0),
        }
    return create_torch_aug_wrapper(Shift, config_func)(x, args, sr, audio_path)


def time_inversion(x: np.ndarray, args: Dict[str, Any], sr: int = 16000, audio_path: Optional[str] = None) -> np.ndarray:
    """Time inversion augmentation wrapper."""
    def config_func(args, sr, audio_path):
        return {'p': args.get('p', 1.0)}
    return create_torch_aug_wrapper(TimeInversion, config_func)(x, args, sr, audio_path)


def add_background_noise(x: np.ndarray, args: Dict[str, Any], sr: int = 16000, audio_path: Optional[str] = None) -> np.ndarray:
    """Add background noise augmentation wrapper."""
    if not TORCH_AUDIO_AVAILABLE:
        raise ImportError("torch-audiomentations is not available")
    
    noise_path = args.get('noise_path', None)
    if noise_path is None:
        raise ValueError("noise_path is required for AddBackgroundNoise")
    
    min_snr = args.get('min_snr_in_db', 3.0)
    max_snr = args.get('max_snr_in_db', 30.0)
    p = args.get('p', 1.0)
    
    # Create augmentation instance - background_paths is positional
    augment = AddBackgroundNoise(
        background_paths=noise_path,
        min_snr_in_db=min_snr,
        max_snr_in_db=max_snr,
        p=p,
        output_type='dict'
    )
    
    # Convert numpy to torch format
    x_torch = numpy_to_torch_audio(x, sr)
    
    # Apply augmentation
    result = augment(samples=x_torch, sample_rate=sr)
    
    # Handle output
    if isinstance(result, dict):
        result_tensor = result['samples']
    elif isinstance(result, torch.Tensor):
        result_tensor = result
    else:
        result_tensor = result.samples
    
    # Convert back to numpy
    return torch_audio_to_numpy(result_tensor)


def add_colored_noise(x: np.ndarray, args: Dict[str, Any], sr: int = 16000, audio_path: Optional[str] = None) -> np.ndarray:
    """Add colored noise augmentation wrapper."""
    def config_func(args, sr, audio_path):
        return {
            'min_snr_in_db': args.get('min_snr_in_db', 3.0),
            'max_snr_in_db': args.get('max_snr_in_db', 30.0),
            'min_f_decay': args.get('min_f_decay', -2.0),
            'max_f_decay': args.get('max_f_decay', 2.0),
            'p': args.get('p', 1.0),
        }
    return create_torch_aug_wrapper(AddColoredNoise, config_func)(x, args, sr, audio_path)


def apply_impulse_response(x: np.ndarray, args: Dict[str, Any], sr: int = 16000, audio_path: Optional[str] = None) -> np.ndarray:
    """Apply impulse response augmentation wrapper."""
    if not TORCH_AUDIO_AVAILABLE:
        raise ImportError("torch-audiomentations is not available")
    
    ir_path = args.get('ir_path', args.get('rir_path', None))
    if ir_path is None:
        raise ValueError("ir_path or rir_path is required for ApplyImpulseResponse")
    
    # Create augmentation instance - ir_paths is positional
    augment = ApplyImpulseResponse(
        ir_paths=ir_path,
        p=args.get('p', 1.0),
        sample_rate=sr,
        compensate_for_propagation_delay=args.get('compensate_for_propagation_delay', False),
        output_type='dict'
    )
    
    # Convert numpy to torch format
    x_torch = numpy_to_torch_audio(x, sr)
    
    # Apply augmentation
    result = augment(samples=x_torch, sample_rate=sr)
    
    # Handle output
    if isinstance(result, dict):
        result_tensor = result['samples']
    elif isinstance(result, torch.Tensor):
        result_tensor = result
    else:
        result_tensor = result.samples
    
    # Convert back to numpy
    return torch_audio_to_numpy(result_tensor)


def band_pass_filter(x: np.ndarray, args: Dict[str, Any], sr: int = 16000, audio_path: Optional[str] = None) -> np.ndarray:
    """Band pass filter augmentation wrapper."""
    def config_func(args, sr, audio_path):
        return {
            'min_center_freq': args.get('min_center_freq', 200.0),
            'max_center_freq': args.get('max_center_freq', 3000.0),
            'min_bandwidth_fraction': args.get('min_bandwidth_fraction', 0.5),
            'max_bandwidth_fraction': args.get('max_bandwidth_fraction', 1.5),
            'p': args.get('p', 1.0),
        }
    return create_torch_aug_wrapper(BandPassFilter, config_func)(x, args, sr, audio_path)


def band_stop_filter(x: np.ndarray, args: Dict[str, Any], sr: int = 16000, audio_path: Optional[str] = None) -> np.ndarray:
    """Band stop filter augmentation wrapper."""
    def config_func(args, sr, audio_path):
        return {
            'min_center_freq': args.get('min_center_freq', 200.0),
            'max_center_freq': args.get('max_center_freq', 3000.0),
            'min_bandwidth_fraction': args.get('min_bandwidth_fraction', 0.5),
            'max_bandwidth_fraction': args.get('max_bandwidth_fraction', 1.5),
            'p': args.get('p', 1.0),
        }
    return create_torch_aug_wrapper(BandStopFilter, config_func)(x, args, sr, audio_path)


def high_pass_filter(x: np.ndarray, args: Dict[str, Any], sr: int = 16000, audio_path: Optional[str] = None) -> np.ndarray:
    """High pass filter augmentation wrapper."""
    def config_func(args, sr, audio_path):
        return {
            'min_cutoff_freq': args.get('min_cutoff_freq', 20.0),
            'max_cutoff_freq': args.get('max_cutoff_freq', 2400.0),
            'p': args.get('p', 1.0),
        }
    return create_torch_aug_wrapper(HighPassFilter, config_func)(x, args, sr, audio_path)


def low_pass_filter(x: Union[np.ndarray, torch.Tensor], args: Dict[str, Any], sr: int = 16000, audio_path: Optional[str] = None) -> torch.Tensor:
    """
    Low pass filter augmentation wrapper with GPU support.
    
    Note: When used in DataLoader workers (num_workers > 0), augmentation will run on CPU.
    To use GPU acceleration, apply augmentation in collate_fn or model forward pass instead.
    
    Args:
        x: numpy array or torch tensor containing audio waveform
        args: dictionary containing configuration. Can include 'device' key to specify device.
        sr: sample rate
        audio_path: path to audio file (for caching if needed)
        
    Returns:
        torch.Tensor containing augmented audio waveform (on specified device, or CPU if in DataLoader worker)
    """
    if not TORCH_AUDIO_AVAILABLE:
        raise ImportError("torch-audiomentations is not available")
    
    # Determine device - prefer GPU if available and not in DataLoader worker
    # Note: In DataLoader workers, GPU is not available, so this will use CPU
    device = args.get('device', None)
    if device is None:
        # Try to use GPU if available, but will fallback to CPU in workers
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Convert input to tensor if needed
    if isinstance(x, np.ndarray):
        x_tensor = torch.from_numpy(x.copy()).float()
    elif isinstance(x, torch.Tensor):
        x_tensor = x.float()
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")
    
    # Ensure tensor is on the correct device
    x_tensor = x_tensor.to(device)
    
    # Prepare tensor format: (samples,) -> (batch, channels, samples)
    if x_tensor.ndim == 1:
        x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
    elif x_tensor.ndim == 2:
        if x_tensor.shape[0] == 1:
            x_tensor = x_tensor.unsqueeze(0)  # (1, 1, samples)
        else:
            x_tensor = x_tensor.unsqueeze(0)  # (1, channels, samples)
    # If already 3D, assume it's (batch, channels, samples)
    
    # Create augmentation configuration
    aug_config = {
        'min_cutoff_freq': args.get('min_cutoff_freq', 150.0),
        'max_cutoff_freq': args.get('max_cutoff_freq', 7500.0),
        'p': args.get('p', 1.0),
        'sample_rate': sr,
        'output_type': 'dict',
    }
    
    # Create augmentation instance and move to device
    augment = LowPassFilter(**aug_config)
    augment = augment.to(device)
    
    # Apply augmentation on GPU
    result = augment(samples=x_tensor, sample_rate=sr)
    
    # Handle output
    if isinstance(result, dict):
        result_tensor = result['samples']
    elif isinstance(result, torch.Tensor):
        result_tensor = result
    else:
        result_tensor = result.samples
    
    # Remove batch and channel dimensions: (1, 1, samples) -> (samples,)
    if result_tensor.ndim == 3:
        result_tensor = result_tensor.squeeze(0).squeeze(0)
    elif result_tensor.ndim == 2:
        result_tensor = result_tensor.squeeze(0)
    
    # Return tensor (keep on GPU)
    return result_tensor


def pitch_shift(x: np.ndarray, args: Dict[str, Any], sr: int = 16000, audio_path: Optional[str] = None) -> np.ndarray:
    """Pitch shift augmentation wrapper."""
    def config_func(args, sr, audio_path):
        return {
            'min_transpose_semitones': args.get('min_transpose_semitones', -4.0),
            'max_transpose_semitones': args.get('max_transpose_semitones', 4.0),
            'p': args.get('p', 1.0),
            'mode': args.get('mode', 'per_example'),
        }
    return create_torch_aug_wrapper(PitchShift, config_func)(x, args, sr, audio_path)


# ============================================================================
# Export all wrapper functions
# ============================================================================

TORCH_AUG_FUNCTIONS = {
    'gain': gain,
    'polarity_inversion': polarity_inversion,
    'peak_normalization': peak_normalization,
    'shift': shift,
    'time_inversion': time_inversion,
    'add_background_noise': add_background_noise,
    'add_colored_noise': add_colored_noise,
    'apply_impulse_response': apply_impulse_response,
    'band_pass_filter': band_pass_filter,
    'band_stop_filter': band_stop_filter,
    'high_pass_filter': high_pass_filter,
    'low_pass_filter': low_pass_filter,
    'pitch_shift': pitch_shift,
}

__all__ = [
    'numpy_to_torch_audio',
    'torch_audio_to_numpy',
    'create_torch_aug_wrapper',
    'TORCH_AUG_FUNCTIONS',
] + list(TORCH_AUG_FUNCTIONS.keys())

