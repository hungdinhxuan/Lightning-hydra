#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Preprocessing Script for Experiment #3: Controlled LPF + Resample (16k→8k)

This script processes audio files with telephony simulation for deepfake detection evaluation.

Usage:
------
    # Process single file
    python preprocess_resample.py --input audio.wav --output output/ --mode lpf_then_resample --fc 3400
    
    # Process folder
    python preprocess_resample.py --input audio_folder/ --output output/ --mode lpf_then_resample --fc 3400
    
    # Run all fc values (sweep)
    python preprocess_resample.py --input audio_folder/ --output output/ --mode lpf_then_resample --sweep
    
    # Direct resample (baseline)
    python preprocess_resample.py --input audio_folder/ --output output/ --mode direct_resample

Author: AI Agent for DSP Research
Date: 2026-01-18
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional
import time

import torch
import torchaudio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.components.telephony_simulation import (
    process_audio,
    sanity_check,
    aliasing_test,
    get_module_info,
    VALID_CUTOFF_FREQUENCIES,
    ModeType,
)


def get_output_filename(
    input_path: Path,
    mode: str,
    fc: Optional[int],
    target_sr: int = 8000,
    output_format: str = 'wav',
) -> str:
    """Generate output filename with mode and fc suffix.
    
    Note: Always outputs as .wav for maximum compatibility with float32 audio.
    """
    stem = input_path.stem
    # Always use .wav extension for output (float32 compatible)
    suffix = f".{output_format}"
    
    if mode == "direct_resample":
        return f"{stem}_direct_res{target_sr // 1000}k{suffix}"
    elif mode == "lpf_then_resample":
        return f"{stem}_lpf{fc}_res{target_sr // 1000}k{suffix}"
    elif mode == "bandpass_then_resample":
        return f"{stem}_bpf300_3400_res{target_sr // 1000}k{suffix}"
    else:
        return f"{stem}_{mode}_res{target_sr // 1000}k{suffix}"


def process_file(
    input_path: Path,
    output_dir: Path,
    mode: ModeType,
    fc: Optional[int] = None,
    target_sr: int = 8000,
    run_sanity_check: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Process a single audio file.
    
    Args:
        input_path: Path to input WAV file
        output_dir: Output directory
        mode: Processing mode
        fc: Cutoff frequency (for lpf_then_resample)
        target_sr: Target sample rate
        run_sanity_check: Whether to run sanity checks
        verbose: Verbose output
        
    Returns:
        Dictionary with processing results
    """
    result = {
        'input': str(input_path),
        'status': 'pending',
        'error': None,
    }
    
    try:
        # Load audio
        waveform, sr = torchaudio.load(str(input_path))
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Ensure 16 kHz (resample if necessary)
        if sr != 16000:
            if verbose:
                print(f"  Resampling input from {sr} Hz to 16000 Hz")
            resampler_in = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler_in(waveform)
            sr = 16000
        
        # Flatten for processing
        waveform_flat = waveform.squeeze(0)
        
        # Process
        start_time = time.time()
        output = process_audio(waveform_flat, sr, mode, fc, target_sr)
        process_time = time.time() - start_time
        
        # Run sanity check if requested
        if run_sanity_check:
            check_results = sanity_check(waveform_flat, output, sr, target_sr, fc, verbose=verbose)
            result['sanity_check'] = check_results
        
        # Generate output filename (always .wav for float32 compatibility)
        output_filename = get_output_filename(input_path, mode, fc, target_sr, output_format='wav')
        output_path = output_dir / output_filename
        
        # Save as WAV (16-bit PCM for maximum compatibility)
        # Convert float32 [-1, 1] to int16 range
        output_2d = output.unsqueeze(0)  # [1, T]
        
        # Clamp to [-1, 1] to prevent clipping artifacts
        output_2d = torch.clamp(output_2d, -1.0, 1.0)
        
        torchaudio.save(
            str(output_path),
            output_2d,
            target_sr,
            encoding='PCM_S',  # Signed PCM (int16)
            bits_per_sample=16,
        )
        
        result.update({
            'status': 'success',
            'output': str(output_path),
            'input_samples': len(waveform_flat),
            'output_samples': len(output),
            'process_time_ms': process_time * 1000,
            'mode': mode,
            'fc': fc,
            'target_sr': target_sr,
        })
        
    except Exception as e:
        result.update({
            'status': 'error',
            'error': str(e),
        })
        if verbose:
            import traceback
            traceback.print_exc()
    
    return result


def find_audio_files(input_path: Path) -> List[Path]:
    """Find all audio files in a directory or return single file."""
    audio_extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}
    
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        files = []
        for ext in audio_extensions:
            files.extend(input_path.glob(f'**/*{ext}'))
            files.extend(input_path.glob(f'**/*{ext.upper()}'))
        return sorted(files)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch audio preprocessing for Experiment #3: Controlled LPF + Resample',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process with LPF @ 3400 Hz then resample to 8 kHz
    python preprocess_resample.py -i audio/ -o output/ -m lpf_then_resample -f 3400
    
    # Sweep all fc values
    python preprocess_resample.py -i audio/ -o output/ -m lpf_then_resample --sweep
    
    # Direct resample (baseline)
    python preprocess_resample.py -i audio/ -o output/ -m direct_resample
    
    # Telephony bandpass
    python preprocess_resample.py -i audio/ -o output/ -m bandpass_then_resample
    
    # Run sanity checks
    python preprocess_resample.py -i audio/ -o output/ -m lpf_then_resample -f 3400 --sanity-check
    
    # Run aliasing test only
    python preprocess_resample.py --aliasing-test -m lpf_then_resample -f 3400
        """
    )
    
    # Input/Output
    parser.add_argument('-i', '--input', type=str, help='Input file or folder path')
    parser.add_argument('-o', '--output', type=str, help='Output folder path')
    
    # Mode
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['direct_resample', 'lpf_then_resample', 'bandpass_then_resample'],
        default='lpf_then_resample',
        help='Processing mode (default: lpf_then_resample)'
    )
    
    # Cutoff frequency
    parser.add_argument(
        '-f', '--fc',
        type=int,
        choices=VALID_CUTOFF_FREQUENCIES,
        help=f'Cutoff frequency in Hz (valid: {VALID_CUTOFF_FREQUENCIES})'
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Process with all fc values (sweep mode)'
    )
    
    # Options
    parser.add_argument('--target-sr', type=int, default=8000, help='Target sample rate (default: 8000)')
    parser.add_argument('--sanity-check', action='store_true', help='Run sanity checks')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--save-log', type=str, help='Save processing log to JSON file')
    
    # Test modes
    parser.add_argument('--aliasing-test', action='store_true', help='Run aliasing test only')
    parser.add_argument('--info', action='store_true', help='Print module info and exit')
    
    # Determinism
    parser.add_argument('--threads', type=int, default=1, help='Number of torch threads (default: 1 for determinism)')
    
    args = parser.parse_args()
    
    # Set torch threads for determinism
    torch.set_num_threads(args.threads)
    if args.verbose:
        print(f"[Config] Torch threads: {args.threads}")
    
    # Print module info
    if args.info:
        print("\n" + "=" * 60)
        print("Telephony Simulation Module Info")
        print("=" * 60)
        info = get_module_info()
        print(json.dumps(info, indent=2))
        return 0
    
    # Run aliasing test
    if args.aliasing_test:
        print("\n" + "=" * 60)
        print("Running Aliasing Test")
        print("=" * 60)
        
        if args.mode == 'lpf_then_resample' and args.fc is None:
            args.fc = 3400  # Default for test
        
        aliasing_test(args.mode, args.fc, verbose=True)
        return 0
    
    # Validate arguments for processing mode
    if args.input is None:
        parser.error("--input is required for processing")
    if args.output is None:
        parser.error("--output is required for processing")
    
    if args.mode == 'lpf_then_resample':
        if args.fc is None and not args.sweep:
            parser.error("--fc or --sweep is required for lpf_then_resample mode")
    
    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    # Find audio files
    audio_files = find_audio_files(input_path)
    if not audio_files:
        print(f"No audio files found in: {input_path}")
        return 1
    
    print(f"\nFound {len(audio_files)} audio file(s)")
    
    # Determine fc values to process
    if args.sweep:
        fc_values = VALID_CUTOFF_FREQUENCIES if args.mode == 'lpf_then_resample' else [None]
    else:
        fc_values = [args.fc]
    
    # Process files
    all_results = []
    total_files = len(audio_files) * len(fc_values)
    processed = 0
    errors = 0
    
    print(f"\nProcessing {total_files} file(s)...")
    print(f"Mode: {args.mode}")
    if args.mode == 'lpf_then_resample':
        print(f"Cutoff frequencies: {fc_values}")
    print(f"Target sample rate: {args.target_sr} Hz")
    print()
    
    for fc in fc_values:
        # Create output subdirectory for this fc
        if args.sweep and len(fc_values) > 1:
            if fc is not None:
                sub_output_dir = output_dir / f"fc_{fc}"
            else:
                sub_output_dir = output_dir / args.mode
        else:
            sub_output_dir = output_dir
        
        sub_output_dir.mkdir(parents=True, exist_ok=True)
        
        for audio_file in audio_files:
            processed += 1
            
            if args.verbose:
                print(f"[{processed}/{total_files}] Processing: {audio_file.name}")
                if fc is not None:
                    print(f"  fc: {fc} Hz")
            else:
                # Progress indicator
                print(f"\r[{processed}/{total_files}] {audio_file.name[:40]}...", end='', flush=True)
            
            result = process_file(
                audio_file,
                sub_output_dir,
                args.mode,
                fc,
                args.target_sr,
                args.sanity_check,
                args.verbose,
            )
            
            all_results.append(result)
            
            if result['status'] == 'error':
                errors += 1
                if not args.verbose:
                    print(f"\n  Error: {result['error']}")
            elif args.verbose:
                print(f"  -> {result['output']}")
                print(f"  Time: {result['process_time_ms']:.1f} ms")
    
    # Summary
    print("\n\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files: {total_files}")
    print(f"Successful: {total_files - errors}")
    print(f"Errors: {errors}")
    print(f"Output directory: {output_dir}")
    
    # Save log
    if args.save_log:
        log_path = Path(args.save_log)
        log_data = {
            'module_info': get_module_info(),
            'config': {
                'mode': args.mode,
                'fc_values': fc_values,
                'target_sr': args.target_sr,
                'input': str(input_path),
                'output': str(output_dir),
            },
            'results': all_results,
            'summary': {
                'total': total_files,
                'success': total_files - errors,
                'errors': errors,
            },
        }
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"\nProcessing log saved to: {log_path}")
    
    return 0 if errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
