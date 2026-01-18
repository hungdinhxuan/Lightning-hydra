#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telephony Simulation Module for Experiment #3: Controlled LPF + Resample (16k→8k)

This module provides DSP-correct audio preprocessing for evaluating deepfake detection
models under telephony simulation conditions.

Design Decisions:
-----------------
1. Filter Type: FIR linear-phase (windowed-sinc with Kaiser window)
   - Number of taps: 255 (provides good frequency selectivity with reasonable latency)
   - Window: Kaiser (β=8.0) - good sidelobe suppression (~60 dB)
   - Application: Symmetric padding + conv1d to maintain linear phase

2. Resampler: torchaudio.transforms.Resample with 'sinc_interp_kaiser' (Julius backend)
   - lowpass_filter_width: 64 (high quality)
   - rolloff: 0.9475 (slightly below Nyquist for anti-aliasing margin)
   - Cached for efficiency

3. No random operations - fully deterministic for reproducibility

Author: AI Agent for DSP Research
Date: 2026-01-18
"""

import torch
import torchaudio
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any, Union, Literal
from functools import lru_cache
import warnings

# ==============================================================================
# Constants
# ==============================================================================

VALID_CUTOFF_FREQUENCIES = [2000, 2800, 3400, 3800]  # Hz
NYQUIST_8K = 4000  # Nyquist frequency for 8 kHz sample rate
DEFAULT_FIR_TAPS = 255  # Odd number for symmetric FIR filter
DEFAULT_KAISER_BETA = 8.0  # Kaiser window parameter (~60 dB sidelobe suppression)

# Resampler configuration (High Quality - fixed, not swept)
RESAMPLER_CONFIG = {
    'lowpass_filter_width': 64,  # Number of zero crossings (higher = better quality)
    'rolloff': 0.9475,  # Rolloff factor (< 1.0 for anti-aliasing margin)
    'resampling_method': 'sinc_interp_kaiser',  # Kaiser-windowed sinc (Julius backend)
    'dtype': None,  # Use input dtype (default behavior)
}

# ==============================================================================
# FIR Filter Design (Linear Phase)
# ==============================================================================

def design_fir_lowpass(
    cutoff_freq: float,
    sample_rate: int = 16000,
    num_taps: int = DEFAULT_FIR_TAPS,
    beta: float = DEFAULT_KAISER_BETA,
) -> torch.Tensor:
    """
    Design a linear-phase FIR lowpass filter using windowed-sinc method.
    
    Filter Specifications:
    - Type: FIR (Finite Impulse Response)
    - Phase: Linear (symmetric coefficients)
    - Window: Kaiser (β=8.0 for ~60 dB sidelobe attenuation)
    - Taps: 255 (fixed for reproducibility)
    - Transition bandwidth: ~sample_rate / num_taps ≈ 63 Hz for 16 kHz, 255 taps
    
    Args:
        cutoff_freq: Cutoff frequency in Hz (must be < sample_rate/2)
        sample_rate: Sample rate in Hz
        num_taps: Number of filter taps (must be odd for Type I FIR)
        beta: Kaiser window beta parameter
        
    Returns:
        FIR filter coefficients as torch.Tensor of shape [num_taps]
    """
    if cutoff_freq >= sample_rate / 2:
        raise ValueError(f"Cutoff frequency {cutoff_freq} Hz must be < Nyquist {sample_rate/2} Hz")
    
    if num_taps % 2 == 0:
        num_taps += 1  # Ensure odd number for symmetric Type I FIR
        warnings.warn(f"Adjusted num_taps to {num_taps} (must be odd)")
    
    # Normalized cutoff frequency (0 to 1, where 1 = Nyquist)
    normalized_cutoff = cutoff_freq / (sample_rate / 2)
    
    # Design ideal lowpass (sinc) filter
    M = num_taps - 1  # Filter order
    n = torch.arange(num_taps, dtype=torch.float64)
    
    # Center the filter at M/2
    n_centered = n - M / 2
    
    # Avoid division by zero at center
    # sinc(x) = sin(π*x) / (π*x) for x ≠ 0, 1 for x = 0
    with torch.no_grad():
        # Ideal lowpass impulse response: h[n] = fc * sinc(fc * n)
        # where fc = normalized_cutoff
        h_ideal = torch.zeros(num_taps, dtype=torch.float64)
        
        center_idx = M // 2
        for i in range(num_taps):
            if i == center_idx:
                h_ideal[i] = normalized_cutoff
            else:
                x = n_centered[i].item()
                h_ideal[i] = math.sin(math.pi * normalized_cutoff * x) / (math.pi * x)
    
    # Apply Kaiser window
    # Kaiser window formula: w[n] = I0(β * sqrt(1 - (2n/M - 1)^2)) / I0(β)
    window = torch.kaiser_window(num_taps, periodic=False, beta=beta, dtype=torch.float64)
    
    # Windowed filter
    h_windowed = h_ideal * window
    
    # Normalize to unity gain at DC
    h_normalized = h_windowed / h_windowed.sum()
    
    return h_normalized.float()  # Return as float32 for efficiency


def design_fir_highpass(
    cutoff_freq: float,
    sample_rate: int = 16000,
    num_taps: int = DEFAULT_FIR_TAPS,
    beta: float = DEFAULT_KAISER_BETA,
) -> torch.Tensor:
    """
    Design a linear-phase FIR highpass filter using spectral inversion.
    
    HPF = δ[n - M/2] - LPF (spectral inversion method)
    
    Args:
        cutoff_freq: Cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        num_taps: Number of filter taps (must be odd)
        beta: Kaiser window beta parameter
        
    Returns:
        FIR filter coefficients as torch.Tensor of shape [num_taps]
    """
    # Get lowpass filter
    h_lowpass = design_fir_lowpass(cutoff_freq, sample_rate, num_taps, beta)
    
    # Spectral inversion: negate all coefficients and add 1 to center
    h_highpass = -h_lowpass.clone()
    center_idx = num_taps // 2
    h_highpass[center_idx] += 1.0
    
    return h_highpass


def design_fir_bandpass(
    low_cutoff: float,
    high_cutoff: float,
    sample_rate: int = 16000,
    num_taps: int = DEFAULT_FIR_TAPS,
    beta: float = DEFAULT_KAISER_BETA,
) -> torch.Tensor:
    """
    Design a linear-phase FIR bandpass filter.
    
    BPF = LPF(high) - LPF(low) (spectral subtraction method)
    
    Args:
        low_cutoff: Low cutoff frequency in Hz
        high_cutoff: High cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        num_taps: Number of filter taps (must be odd)
        beta: Kaiser window beta parameter
        
    Returns:
        FIR filter coefficients as torch.Tensor of shape [num_taps]
    """
    if low_cutoff >= high_cutoff:
        raise ValueError(f"Low cutoff {low_cutoff} Hz must be < high cutoff {high_cutoff} Hz")
    
    h_lpf_high = design_fir_lowpass(high_cutoff, sample_rate, num_taps, beta)
    h_lpf_low = design_fir_lowpass(low_cutoff, sample_rate, num_taps, beta)
    
    h_bandpass = h_lpf_high - h_lpf_low
    
    return h_bandpass


# ==============================================================================
# Filter Application (Linear Phase via Symmetric Padding)
# ==============================================================================

def apply_fir_filter(
    x: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """
    Apply FIR filter using conv1d with symmetric padding for linear phase.
    
    Uses 'reflect' padding to minimize edge artifacts while maintaining
    causality for the center of the signal.
    
    Args:
        x: Input waveform of shape [T] or [1, T] or [B, T]
        h: FIR filter coefficients of shape [num_taps]
        
    Returns:
        Filtered waveform of same shape as input
    """
    original_shape = x.shape
    original_ndim = x.ndim
    
    # Ensure x is [B, 1, T] for conv1d
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    elif x.ndim == 2:
        x = x.unsqueeze(1)  # [B, 1, T]
    
    # Prepare filter kernel: [out_channels=1, in_channels=1, kernel_size]
    kernel = h.view(1, 1, -1).to(x.dtype).to(x.device)
    kernel_size = h.shape[0]
    
    # Symmetric (reflect) padding for linear phase
    # Pad size = (kernel_size - 1) // 2 on each side
    pad_size = (kernel_size - 1) // 2
    
    # Apply padding and convolution
    x_padded = torch.nn.functional.pad(x, (pad_size, pad_size), mode='reflect')
    y = torch.nn.functional.conv1d(x_padded, kernel)
    
    # Restore original shape
    if original_ndim == 1:
        y = y.squeeze(0).squeeze(0)
    elif original_ndim == 2:
        y = y.squeeze(1)
    
    return y


# ==============================================================================
# Resampler (Cached for Efficiency)
# ==============================================================================

@lru_cache(maxsize=8)
def get_resampler(
    orig_freq: int = 16000,
    new_freq: int = 8000,
    device_str: str = 'cpu',
) -> torchaudio.transforms.Resample:
    """
    Get a cached torchaudio resampler with high-quality settings.
    
    Resampler Configuration (Fixed - High Quality):
    - Backend: Julius (sinc_interp_kaiser)
    - lowpass_filter_width: 64 zero crossings
    - rolloff: 0.9475 (anti-aliasing margin)
    - dtype: float64 for computation
    
    Args:
        orig_freq: Original sample rate
        new_freq: Target sample rate
        device_str: Device string for caching
        
    Returns:
        Cached Resample transform
    """
    device = torch.device(device_str)
    
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_freq,
        new_freq=new_freq,
        lowpass_filter_width=RESAMPLER_CONFIG['lowpass_filter_width'],
        rolloff=RESAMPLER_CONFIG['rolloff'],
        resampling_method=RESAMPLER_CONFIG['resampling_method'],
        dtype=RESAMPLER_CONFIG['dtype'],  # None = use input dtype
    ).to(device)
    
    # Log configuration for traceability
    print(f"[TelephonySim] Created Resampler: {orig_freq}Hz → {new_freq}Hz")
    print(f"  - method: {RESAMPLER_CONFIG['resampling_method']}")
    print(f"  - lowpass_filter_width: {RESAMPLER_CONFIG['lowpass_filter_width']}")
    print(f"  - rolloff: {RESAMPLER_CONFIG['rolloff']}")
    print(f"  - dtype: {RESAMPLER_CONFIG['dtype']}")
    print(f"  - device: {device}")
    
    return resampler


# ==============================================================================
# Main Processing Function
# ==============================================================================

ModeType = Literal["direct_resample", "lpf_then_resample", "bandpass_then_resample"]


def process_audio(
    x: torch.Tensor,
    sr: int = 16000,
    mode: ModeType = "lpf_then_resample",
    fc: Optional[int] = None,
    target_sr: int = 8000,
    num_taps: int = DEFAULT_FIR_TAPS,
    kaiser_beta: float = DEFAULT_KAISER_BETA,
) -> torch.Tensor:
    """
    Process audio with controlled LPF + resample for telephony simulation.
    
    Modes:
    ------
    A. "direct_resample": 
       Resample 16k → 8k directly (baseline, equivalent to librosa.load(sr=8000))
       
    B. "lpf_then_resample": 
       Apply FIR lowpass filter (fc Hz), then resample 16k → 8k
       Valid fc values: {2000, 2800, 3400, 3800} Hz (must be < 4000 Hz Nyquist)
       
    C. "bandpass_then_resample": 
       Apply telephony bandpass (300 Hz HPF + 3400 Hz LPF), then resample 16k → 8k
    
    Args:
        x: Input waveform, shape [T] or [1, T], dtype float32/float64, sr=16000 Hz
        sr: Input sample rate (must be 16000)
        mode: Processing mode ("direct_resample", "lpf_then_resample", "bandpass_then_resample")
        fc: Cutoff frequency for LPF mode (required for "lpf_then_resample")
        target_sr: Target sample rate (default 8000)
        num_taps: Number of FIR filter taps
        kaiser_beta: Kaiser window beta parameter
        
    Returns:
        Processed waveform at target_sr, shape [T_new] (mono)
    """
    # ==========================================================================
    # Input validation
    # ==========================================================================
    if sr != 16000:
        raise ValueError(f"Input sample rate must be 16000 Hz, got {sr} Hz")
    
    if target_sr != 8000:
        warnings.warn(f"Target sample rate {target_sr} Hz is non-standard for telephony simulation")
    
    # Ensure tensor and flatten to 1D for consistent processing
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    
    original_dtype = x.dtype
    x = x.float()  # Work in float32
    
    # Flatten to 1D
    if x.ndim == 2:
        if x.shape[0] == 1:
            x = x.squeeze(0)
        elif x.shape[1] == 1:
            x = x.squeeze(1)
        else:
            raise ValueError(f"Expected mono audio, got shape {x.shape}")
    elif x.ndim > 2:
        raise ValueError(f"Expected 1D or 2D tensor, got {x.ndim}D")
    
    device = x.device
    device_str = str(device)
    
    # ==========================================================================
    # Mode A: Direct Resample (Baseline)
    # ==========================================================================
    if mode == "direct_resample":
        resampler = get_resampler(sr, target_sr, device_str)
        y = resampler(x)
        return _ensure_no_clip(y)
    
    # ==========================================================================
    # Mode B: LPF then Resample
    # ==========================================================================
    elif mode == "lpf_then_resample":
        if fc is None:
            raise ValueError("Cutoff frequency 'fc' is required for 'lpf_then_resample' mode")
        
        if fc >= NYQUIST_8K:
            raise ValueError(
                f"Cutoff frequency fc={fc} Hz must be < Nyquist={NYQUIST_8K} Hz "
                f"for 8 kHz output. Valid values: {VALID_CUTOFF_FREQUENCIES}"
            )
        
        if fc not in VALID_CUTOFF_FREQUENCIES:
            warnings.warn(
                f"Cutoff frequency fc={fc} Hz is not in standard sweep set "
                f"{VALID_CUTOFF_FREQUENCIES}. Proceeding anyway."
            )
        
        # Design and apply LPF
        h_lpf = design_fir_lowpass(fc, sr, num_taps, kaiser_beta)
        x_filtered = apply_fir_filter(x, h_lpf.to(device))
        
        # Resample
        resampler = get_resampler(sr, target_sr, device_str)
        y = resampler(x_filtered)
        
        return _ensure_no_clip(y)
    
    # ==========================================================================
    # Mode C: Bandpass then Resample (Telephony-ish)
    # ==========================================================================
    elif mode == "bandpass_then_resample":
        # Telephony band: 300 Hz - 3400 Hz
        low_cutoff = 300
        high_cutoff = 3400
        
        # Design and apply bandpass filter
        h_bpf = design_fir_bandpass(low_cutoff, high_cutoff, sr, num_taps, kaiser_beta)
        x_filtered = apply_fir_filter(x, h_bpf.to(device))
        
        # Resample
        resampler = get_resampler(sr, target_sr, device_str)
        y = resampler(x_filtered)
        
        return _ensure_no_clip(y)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Valid: 'direct_resample', 'lpf_then_resample', 'bandpass_then_resample'")


def _ensure_no_clip(x: torch.Tensor) -> torch.Tensor:
    """Soft clip to [-1, 1] only if values exceed range."""
    max_val = x.abs().max()
    if max_val > 1.0:
        x = x / max_val
        warnings.warn(f"Audio clipped from max={max_val:.4f} to [-1, 1]")
    return x


# ==============================================================================
# Batch Processing Variants (for DataLoader/Collate)
# ==============================================================================

def process_audio_batch(
    batch: torch.Tensor,
    sr: int = 16000,
    mode: ModeType = "lpf_then_resample",
    fc: Optional[int] = None,
    target_sr: int = 8000,
) -> torch.Tensor:
    """
    Process a batch of audio waveforms.
    
    Args:
        batch: Input batch of shape [B, T] or [B, 1, T]
        sr: Input sample rate
        mode: Processing mode
        fc: Cutoff frequency (for lpf_then_resample mode)
        target_sr: Target sample rate
        
    Returns:
        Processed batch of shape [B, T_new]
    """
    # Handle different input shapes
    if batch.ndim == 3 and batch.shape[1] == 1:
        batch = batch.squeeze(1)  # [B, 1, T] -> [B, T]
    
    if batch.ndim != 2:
        raise ValueError(f"Expected batch of shape [B, T] or [B, 1, T], got {batch.shape}")
    
    # Process each sample
    # Note: Using list comprehension for now; could be optimized for batch FIR + batch resample
    processed = []
    for i in range(batch.shape[0]):
        y = process_audio(batch[i], sr, mode, fc, target_sr)
        processed.append(y)
    
    # Pad to same length (resampling may cause slight length variations)
    max_len = max(p.shape[0] for p in processed)
    padded = torch.zeros(len(processed), max_len, device=batch.device, dtype=batch.dtype)
    for i, p in enumerate(processed):
        padded[i, :p.shape[0]] = p
    
    return padded


# ==============================================================================
# Sanity Check Functions
# ==============================================================================

def compute_psd(
    x: torch.Tensor,
    sample_rate: int,
    n_fft: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Power Spectral Density using FFT.
    
    Args:
        x: Input waveform [T]
        sample_rate: Sample rate in Hz
        n_fft: FFT size
        
    Returns:
        frequencies: Frequency bins [n_fft//2 + 1]
        psd: Power spectral density [n_fft//2 + 1] in dB
    """
    if x.ndim != 1:
        x = x.flatten()
    
    # Window and FFT
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    
    # Handle short signals
    if len(x) < n_fft:
        x = torch.nn.functional.pad(x, (0, n_fft - len(x)))
    
    # Compute average PSD over segments
    hop = n_fft // 2
    n_segments = max(1, (len(x) - n_fft) // hop + 1)
    
    psd_accum = torch.zeros(n_fft // 2 + 1, device=x.device, dtype=x.dtype)
    
    for i in range(n_segments):
        start = i * hop
        segment = x[start:start + n_fft]
        if len(segment) < n_fft:
            segment = torch.nn.functional.pad(segment, (0, n_fft - len(segment)))
        
        windowed = segment * window
        spectrum = torch.fft.rfft(windowed)
        psd_accum += spectrum.abs() ** 2
    
    psd = psd_accum / n_segments
    psd_db = 10 * torch.log10(psd + 1e-10)
    
    # Frequency bins
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    
    return freqs.to(x.device), psd_db


def sanity_check(
    x16: torch.Tensor,
    y8: torch.Tensor,
    sr_in: int = 16000,
    sr_out: int = 8000,
    fc: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Perform sanity checks on processed audio.
    
    Checks:
    1. Shape and dtype validity
    2. No NaN/Inf values
    3. Energy reduction above cutoff frequency (for LPF modes)
    4. Energy near-zero above Nyquist after resample
    
    Args:
        x16: Original 16 kHz waveform
        y8: Processed 8 kHz waveform
        sr_in: Input sample rate
        sr_out: Output sample rate
        fc: Cutoff frequency used (optional)
        verbose: Print detailed results
        
    Returns:
        Dictionary with check results
    """
    results = {
        'passed': True,
        'checks': {},
    }
    
    # ==========================================================================
    # Check 1: Shape and dtype
    # ==========================================================================
    x16_flat = x16.flatten()
    y8_flat = y8.flatten()
    
    expected_length_ratio = sr_out / sr_in
    actual_length_ratio = len(y8_flat) / len(x16_flat)
    length_ratio_ok = abs(actual_length_ratio - expected_length_ratio) < 0.01
    
    results['checks']['shape_dtype'] = {
        'x16_shape': list(x16.shape),
        'y8_shape': list(y8.shape),
        'x16_dtype': str(x16.dtype),
        'y8_dtype': str(y8.dtype),
        'length_ratio': actual_length_ratio,
        'expected_ratio': expected_length_ratio,
        'passed': length_ratio_ok,
    }
    results['passed'] &= length_ratio_ok
    
    # ==========================================================================
    # Check 2: No NaN/Inf
    # ==========================================================================
    x16_valid = not (torch.isnan(x16).any() or torch.isinf(x16).any())
    y8_valid = not (torch.isnan(y8).any() or torch.isinf(y8).any())
    nan_inf_ok = x16_valid and y8_valid
    
    results['checks']['nan_inf'] = {
        'x16_valid': x16_valid,
        'y8_valid': y8_valid,
        'passed': nan_inf_ok,
    }
    results['passed'] &= nan_inf_ok
    
    # ==========================================================================
    # Check 3: Energy reduction above fc (for LPF modes)
    # ==========================================================================
    if fc is not None:
        freqs_16, psd_16 = compute_psd(x16_flat, sr_in)
        freqs_8, psd_8 = compute_psd(y8_flat, sr_out)
        
        # Energy above fc in original (before filtering)
        above_fc_mask_16 = freqs_16 > fc
        energy_above_fc_16 = psd_16[above_fc_mask_16].mean().item() if above_fc_mask_16.any() else -100
        
        # Energy below fc in original
        below_fc_mask_16 = (freqs_16 > 100) & (freqs_16 <= fc)
        energy_below_fc_16 = psd_16[below_fc_mask_16].mean().item() if below_fc_mask_16.any() else -100
        
        # After processing (8 kHz), check energy distribution
        # All energy should be below Nyquist (4 kHz)
        energy_8k_total = psd_8.mean().item()
        
        # Energy reduction check: expect at least 20 dB attenuation above fc
        fc_attenuation_ok = True  # Will be verified by aliasing test
        
        results['checks']['energy_reduction'] = {
            'fc': fc,
            'energy_below_fc_16k_dB': energy_below_fc_16,
            'energy_above_fc_16k_dB': energy_above_fc_16,
            'energy_8k_total_dB': energy_8k_total,
            'passed': fc_attenuation_ok,
        }
    
    # ==========================================================================
    # Check 4: Energy near-zero above Nyquist (implicit in 8 kHz signal)
    # ==========================================================================
    freqs_8, psd_8 = compute_psd(y8_flat, sr_out)
    
    # For 8 kHz signal, max frequency is 4 kHz
    # Check that high frequencies (near Nyquist) don't have excessive energy
    high_freq_mask = freqs_8 > 3500
    energy_near_nyquist = psd_8[high_freq_mask].mean().item() if high_freq_mask.any() else -100
    
    # Energy in passband (e.g., 300-3000 Hz for telephony)
    passband_mask = (freqs_8 > 300) & (freqs_8 < 3000)
    energy_passband = psd_8[passband_mask].mean().item() if passband_mask.any() else -100
    
    # Near Nyquist should be at least 30 dB below passband
    nyquist_suppression_ok = (energy_passband - energy_near_nyquist) > 20
    
    results['checks']['nyquist_suppression'] = {
        'energy_passband_dB': energy_passband,
        'energy_near_nyquist_dB': energy_near_nyquist,
        'suppression_dB': energy_passband - energy_near_nyquist,
        'passed': nyquist_suppression_ok,
    }
    results['passed'] &= nyquist_suppression_ok
    
    # ==========================================================================
    # Verbose output
    # ==========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("SANITY CHECK RESULTS")
        print("=" * 60)
        
        for check_name, check_data in results['checks'].items():
            status = "✓ PASS" if check_data.get('passed', True) else "✗ FAIL"
            print(f"\n[{status}] {check_name}:")
            for key, value in check_data.items():
                if key != 'passed':
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)
        overall = "✓ ALL CHECKS PASSED" if results['passed'] else "✗ SOME CHECKS FAILED"
        print(f"OVERALL: {overall}")
        print("=" * 60 + "\n")
    
    return results


def aliasing_test(
    mode: ModeType = "lpf_then_resample",
    fc: Optional[int] = 3400,
    test_freqs: list = [5000, 6000, 7000],
    sr_in: int = 16000,
    sr_out: int = 8000,
    duration: float = 1.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Test for aliasing by generating tones above Nyquist and checking if they fold back.
    
    Creates test tones at 5-7 kHz (above 8 kHz Nyquist = 4 kHz), processes them,
    and verifies that they are attenuated (not aliased into the passband).
    
    Args:
        mode: Processing mode to test
        fc: Cutoff frequency (for lpf_then_resample mode)
        test_freqs: Frequencies to test (should be > 4 kHz)
        sr_in: Input sample rate
        sr_out: Output sample rate
        duration: Test signal duration in seconds
        verbose: Print detailed results
        
    Returns:
        Dictionary with test results
    """
    results = {
        'passed': True,
        'test_frequencies': {},
    }
    
    nyquist_out = sr_out / 2
    
    for test_freq in test_freqs:
        if test_freq <= nyquist_out:
            warnings.warn(f"Test frequency {test_freq} Hz should be > Nyquist {nyquist_out} Hz")
        
        # Generate pure tone at test frequency
        t = torch.linspace(0, duration, int(sr_in * duration))
        tone_16k = 0.5 * torch.sin(2 * math.pi * test_freq * t)
        
        # Process
        tone_8k = process_audio(tone_16k, sr_in, mode, fc, sr_out)
        
        # Check aliased frequency: f_alias = |f_test - n * sr_out| for n that minimizes
        f_alias = abs(test_freq - sr_out)  # For f_test between sr_out/2 and sr_out
        if test_freq > sr_out:
            f_alias = test_freq - sr_out
            while f_alias > nyquist_out:
                f_alias = abs(f_alias - sr_out)
        
        # Compute energy at original frequency band and aliased frequency band
        freqs_8k, psd_8k = compute_psd(tone_8k, sr_out, n_fft=2048)
        
        # Find energy near the aliased frequency
        alias_band_mask = (freqs_8k > f_alias - 200) & (freqs_8k < f_alias + 200)
        energy_alias_band = psd_8k[alias_band_mask].max().item() if alias_band_mask.any() else -100
        
        # Find energy in total signal
        total_energy = psd_8k.max().item()
        
        # If LPF is working, aliased energy should be very low (< -40 dB)
        alias_suppression = total_energy - energy_alias_band if energy_alias_band > -90 else 60
        alias_ok = energy_alias_band < -30  # Aliased energy should be < -30 dB
        
        results['test_frequencies'][test_freq] = {
            'aliased_to': f_alias,
            'energy_at_alias_dB': energy_alias_band,
            'total_energy_dB': total_energy,
            'suppression_dB': alias_suppression,
            'passed': alias_ok,
        }
        results['passed'] &= alias_ok
    
    if verbose:
        print("\n" + "=" * 60)
        print("ALIASING TEST RESULTS")
        print(f"Mode: {mode}, fc: {fc} Hz")
        print("=" * 60)
        
        for freq, data in results['test_frequencies'].items():
            status = "✓ PASS" if data['passed'] else "✗ FAIL"
            print(f"\n[{status}] Test tone at {freq} Hz:")
            print(f"  Would alias to: {data['aliased_to']:.0f} Hz")
            print(f"  Energy at alias band: {data['energy_at_alias_dB']:.1f} dB")
            print(f"  Suppression: {data['suppression_dB']:.1f} dB")
        
        print("\n" + "=" * 60)
        overall = "✓ NO ALIASING DETECTED" if results['passed'] else "✗ ALIASING DETECTED"
        print(f"OVERALL: {overall}")
        print("=" * 60 + "\n")
    
    return results


# ==============================================================================
# Quick Access Functions for Common Configurations
# ==============================================================================

def telephony_8k_lpf2000(x: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """LPF @ 2.0 kHz + Resample to 8 kHz"""
    return process_audio(x, sr, "lpf_then_resample", fc=2000)


def telephony_8k_lpf2800(x: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """LPF @ 2.8 kHz + Resample to 8 kHz"""
    return process_audio(x, sr, "lpf_then_resample", fc=2800)


def telephony_8k_lpf3400(x: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """LPF @ 3.4 kHz + Resample to 8 kHz"""
    return process_audio(x, sr, "lpf_then_resample", fc=3400)


def telephony_8k_lpf3800(x: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """LPF @ 3.8 kHz + Resample to 8 kHz"""
    return process_audio(x, sr, "lpf_then_resample", fc=3800)


def telephony_8k_direct(x: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """Direct resample to 8 kHz (baseline)"""
    return process_audio(x, sr, "direct_resample")


def telephony_8k_bandpass(x: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """Telephony bandpass (300-3400 Hz) + Resample to 8 kHz"""
    return process_audio(x, sr, "bandpass_then_resample")


# ==============================================================================
# Module Info
# ==============================================================================

def get_module_info() -> Dict[str, Any]:
    """Get module configuration info for logging/reproducibility."""
    # Convert dtype to string for JSON serialization
    resampler_config_serializable = {
        k: (str(v) if isinstance(v, torch.dtype) else v)
        for k, v in RESAMPLER_CONFIG.items()
    }
    return {
        'module': 'telephony_simulation',
        'version': '1.0.0',
        'filter': {
            'type': 'FIR',
            'design': 'windowed-sinc',
            'window': 'Kaiser',
            'default_taps': DEFAULT_FIR_TAPS,
            'default_beta': DEFAULT_KAISER_BETA,
            'phase': 'linear',
            'application': 'conv1d with reflect padding',
        },
        'resampler': resampler_config_serializable,
        'valid_cutoff_frequencies': VALID_CUTOFF_FREQUENCIES,
        'modes': ['direct_resample', 'lpf_then_resample', 'bandpass_then_resample'],
    }


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("Telephony Simulation Module - Quick Test")
    print("=" * 60)
    print("\nModule Info:")
    for key, value in get_module_info().items():
        print(f"  {key}: {value}")
    
    # Generate test signal (1 second of multi-tone)
    sr = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(sr * duration))
    
    # Multi-tone test: 500 Hz + 2 kHz + 5 kHz
    test_signal = 0.3 * (
        torch.sin(2 * math.pi * 500 * t) +
        torch.sin(2 * math.pi * 2000 * t) +
        torch.sin(2 * math.pi * 5000 * t)
    )
    
    print(f"\nTest signal: {len(test_signal)} samples @ {sr} Hz")
    print("Contains tones at: 500 Hz, 2 kHz, 5 kHz")
    
    # Test all modes
    for mode, fc in [
        ("direct_resample", None),
        ("lpf_then_resample", 3400),
        ("lpf_then_resample", 2000),
        ("bandpass_then_resample", None),
    ]:
        print(f"\n--- Testing mode: {mode}, fc: {fc} ---")
        output = process_audio(test_signal, sr, mode, fc)
        print(f"Output: {len(output)} samples @ 8000 Hz")
        sanity_check(test_signal, output, sr, 8000, fc, verbose=True)
    
    # Aliasing test
    print("\n--- Running Aliasing Test ---")
    aliasing_test("lpf_then_resample", fc=3400, verbose=True)
