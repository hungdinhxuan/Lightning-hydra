import os
import random
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
import psola
import scipy.signal as sig
import argparse
import json
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

## Source code is referenced from:
# https://github.com/JanWilczek/python-auto-tune/blob/main/auto_tune.py

def parse_arguments():
    """Parse command line arguments for dynamic configuration"""
    parser = argparse.ArgumentParser(description="Generate auto-tuned audio files from protocol")
    
    parser.add_argument("--protocol_path", type=str, required=True,
                       help="Path to the protocol file")
    parser.add_argument("--base_input_dir", type=str, required=True,
                       help="Base directory containing input audio files")
    parser.add_argument("--base_output_dir", type=str, required=True,
                       help="Base directory for output files")
    parser.add_argument("--noise_type", type=str, default="auto_tune",
                       help="Noise type identifier (default: auto_tune)")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of parallel workers (default: 4)")
    parser.add_argument("--sample_rate", type=int, default=16000,
                       help="Target sample rate (default: 16000)")
    parser.add_argument("--use_common_scales", action="store_true",
                       help="Use only common scales instead of comprehensive pool")
    parser.add_argument("--frame_length", type=int, default=2048,
                       help="Frame length for pitch analysis (default: 2048)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--sequential", action="store_true",
                       help="Use sequential processing instead of parallel")
    
    return parser.parse_args()

# --- 스케일 기반 pitch 보정 함수 ---
SEMITONES_IN_OCTAVE = 12

def generate_comprehensive_scale_pool():
    """Generate a comprehensive scale pool covering all librosa supported keys and modes"""
    
    # Basic tonics (natural notes)
    basic_tonics = ["C", "D", "E", "F", "G", "A", "B"]
    
    # Single accidentals
    sharp_tonics = [f"{t}#" for t in basic_tonics]
    flat_tonics = [f"{t}b" for t in basic_tonics]
    
    # Double accidentals (less common but supported)
    double_sharp_tonics = [f"{t}##" for t in ["C", "D", "F", "G", "A"]]  # Avoid E##, B##
    double_flat_tonics = [f"{t}bb" for t in ["D", "E", "G", "A", "B"]]   # Avoid C♭♭, F♭♭
    
    # All tonics combined
    all_tonics = basic_tonics + sharp_tonics + flat_tonics + double_sharp_tonics + double_flat_tonics
    
    # All supported modes and their abbreviations
    modes = [
        "maj", "major",  # Major
        "min",           # Minor
        "ionian", "ion", # Ionian (same as major)
        "dorian", "dor", # Dorian
        "phrygian", "phr", "phryg",  # Phrygian
        "lydian", "lyd", "lydi",     # Lydian
        "mixolydian", "mix", "mixolyd",  # Mixolydian
        "aeolian", "aeo", "aeol",    # Aeolian (same as minor)
        "locrian", "loc", "locr"     # Locrian
    ]
    
    # Generate all combinations
    scale_pool = []
    for tonic in all_tonics:
        for mode in modes:
            scale_pool.append(f"{tonic}:{mode}")
    
    return scale_pool

def generate_common_scale_pool():
    """Generate commonly used scales for practical purposes"""
    common_tonics = ["C", "D", "E", "F", "G", "A", "B", 
                     "C#", "Db", "D#", "Eb", "F#", "Gb", "G#", "Ab", "A#", "Bb"]
    common_modes = ["maj", "min", "dor", "phr", "lyd", "mix", "aeo", "loc"]
    
    scale_pool = []
    for tonic in common_tonics:
        for mode in common_modes:
            scale_pool.append(f"{tonic}:{mode}")
    
    return scale_pool

def degrees_from(scale: str):
    degrees = librosa.key_to_degrees(scale)
    return np.concatenate((degrees, [degrees[0] + SEMITONES_IN_OCTAVE]))

def closest_pitch_from_scale(f0, scale):
    if np.isnan(f0):
        return np.nan
    degrees = degrees_from(scale)
    midi_note = librosa.hz_to_midi(f0)
    degree = midi_note % SEMITONES_IN_OCTAVE
    degree_id = np.argmin(np.abs(degrees - degree))
    midi_note -= (degree - degrees[degree_id])
    return librosa.midi_to_hz(midi_note)

def aclosest_pitch_from_scale(f0, scale):
    sanitized = np.zeros_like(f0)
    for i in range(f0.shape[0]):
        sanitized[i] = closest_pitch_from_scale(f0[i], scale)
    smoothed = sig.medfilt(sanitized, kernel_size=11)
    smoothed[np.isnan(smoothed)] = sanitized[np.isnan(smoothed)]
    return smoothed

def apply_auto_tune(audio, sr, scale="C:maj", frame_length=2048):
    """Apply auto-tune effect to audio with error handling"""
    try:
        hop_length = frame_length // 4
        fmin = librosa.note_to_hz('C2')
        fmax = librosa.note_to_hz('C7')
        
        f0, voiced_flag, _ = librosa.pyin(audio,
                                          frame_length=frame_length,
                                          hop_length=hop_length,
                                          sr=sr,
                                          fmin=fmin,
                                          fmax=fmax)
        
        corrected_f0 = aclosest_pitch_from_scale(f0, scale)
        corrected_f0 = np.where(np.isnan(corrected_f0), f0, corrected_f0)
        
        return psola.vocode(audio, sample_rate=sr, target_pitch=corrected_f0, fmin=fmin, fmax=fmax)
    except Exception as e:
        # If auto-tune fails, return original audio
        print(f"Warning: Auto-tune failed for scale {scale}, returning original audio: {e}")
        return audio

def read_protocol_file(protocol_path):
    """Read protocol file and return list of file info"""
    file_list = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                file_path = parts[0]
                split = parts[1]  # train/test/dev
                label = parts[2]  # spoof/bonafide
                file_list.append({
                    'file_path': file_path,
                    'split': split,
                    'label': label
                })
    return file_list

class ThreadSafeResults:
    """Thread-safe container for results"""
    def __init__(self):
        self.results = []
        self.errors = []
        self.lock = threading.Lock()
    
    def add_result(self, result):
        with self.lock:
            if result['status'] == 'success':
                self.results.append(result)
            else:
                self.errors.append(result)
    
    def get_results(self):
        with self.lock:
            return self.results.copy(), self.errors.copy()

def process_single_file(file_info, base_input_dir, base_output_dir, noise_type, 
                       scale_pool, sample_rate, frame_length, results_container):
    """Process a single audio file with auto-tune"""
    try:
        # Construct full input path
        input_path = os.path.join(base_input_dir, file_info['file_path'])
        
        # Check if input file exists
        if not os.path.exists(input_path):
            result = {'status': 'error', 'file_path': file_info['file_path'], 'error': 'File not found'}
            results_container.add_result(result)
            return
        
        # Load audio with error handling
        try:
            audio, sr = librosa.load(input_path, sr=sample_rate)
        except Exception as e:
            result = {'status': 'error', 'file_path': file_info['file_path'], 'error': f'Failed to load audio: {str(e)}'}
            results_container.add_result(result)
            return
        
        # Skip if audio is too short or has issues
        if len(audio) < sample_rate * 0.1:  # Less than 0.1 seconds
            result = {'status': 'error', 'file_path': file_info['file_path'], 'error': 'Audio too short'}
            results_container.add_result(result)
            return
        
        # Random scale selection (thread-safe)
        scale = random.choice(scale_pool)
        tuned_audio = apply_auto_tune(audio, sr, scale=scale, frame_length=frame_length)
        
        # Create output directory structure
        # Maintain the same directory structure as input
        relative_dir = os.path.dirname(file_info['file_path'])
        output_dir = os.path.join(base_output_dir, relative_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        base_name = os.path.basename(file_info['file_path'])
        name_without_ext = os.path.splitext(base_name)[0]
        ext = os.path.splitext(base_name)[1]
        output_filename = f"{name_without_ext}_{noise_type}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save processed audio
        try:
            sf.write(output_path, tuned_audio, sample_rate)
        except Exception as e:
            result = {'status': 'error', 'file_path': file_info['file_path'], 'error': f'Failed to save audio: {str(e)}'}
            results_container.add_result(result)
            return
        
        # Create metadata entry
        result = {
            'status': 'success',
            'original_path': file_info['file_path'],
            'output_path': os.path.relpath(output_path, base_output_dir),
            'split': file_info['split'],
            'original_label': file_info['label'],
            'new_label': f"{file_info['label']}_{noise_type}",
            'scale_used': scale
        }
        
        results_container.add_result(result)
        
    except Exception as e:
        result = {
            'status': 'error',
            'file_path': file_info['file_path'],
            'error': f"Process error: {str(e)}"
        }
        results_container.add_result(result)

def process_files_sequential(file_list, base_input_dir, base_output_dir, noise_type, 
                           scale_pool, sample_rate, frame_length):
    """Process files sequentially with progress bar"""
    results = []
    errors = []
    
    for file_info in tqdm(file_list, desc="Processing files"):
        try:
            # Construct full input path
            input_path = os.path.join(base_input_dir, file_info['file_path'])
            
            # Check if input file exists
            if not os.path.exists(input_path):
                errors.append({'status': 'error', 'file_path': file_info['file_path'], 'error': 'File not found'})
                continue
            
            # Load audio with error handling
            try:
                audio, sr = librosa.load(input_path, sr=sample_rate)
            except Exception as e:
                errors.append({'status': 'error', 'file_path': file_info['file_path'], 'error': f'Failed to load audio: {str(e)}'})
                continue
            
            # Skip if audio is too short or has issues
            if len(audio) < sample_rate * 0.1:  # Less than 0.1 seconds
                errors.append({'status': 'error', 'file_path': file_info['file_path'], 'error': 'Audio too short'})
                continue
            
            # Random scale selection
            scale = random.choice(scale_pool)
            tuned_audio = apply_auto_tune(audio, sr, scale=scale, frame_length=frame_length)
            
            # Create output directory structure
            relative_dir = os.path.dirname(file_info['file_path'])
            output_dir = os.path.join(base_output_dir, relative_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename
            base_name = os.path.basename(file_info['file_path'])
            name_without_ext = os.path.splitext(base_name)[0]
            ext = os.path.splitext(base_name)[1]
            output_filename = f"{name_without_ext}_{noise_type}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save processed audio
            try:
                sf.write(output_path, tuned_audio, sample_rate)
            except Exception as e:
                errors.append({'status': 'error', 'file_path': file_info['file_path'], 'error': f'Failed to save audio: {str(e)}'})
                continue
            
            # Create metadata entry
            result = {
                'status': 'success',
                'original_path': file_info['file_path'],
                'output_path': os.path.relpath(output_path, base_output_dir),
                'split': file_info['split'],
                'original_label': file_info['label'],
                'new_label': f"{file_info['label']}_{noise_type}",
                'scale_used': scale
            }
            
            results.append(result)
            
        except Exception as e:
            errors.append({
                'status': 'error',
                'file_path': file_info['file_path'],
                'error': f"Process error: {str(e)}"
            })
    
    return results, errors

def main():
    """Main processing function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Generate scale pool
    if args.use_common_scales:
        scale_pool = generate_common_scale_pool()
        print(f"Using common scales pool with {len(scale_pool)} scales")
    else:
        scale_pool = generate_comprehensive_scale_pool()
        print(f"Using comprehensive scales pool with {len(scale_pool)} scales")
    
    # Read protocol file
    print(f"Reading protocol file: {args.protocol_path}")
    if not os.path.exists(args.protocol_path):
        print(f"❌ Error: Protocol file not found: {args.protocol_path}")
        return
    
    file_list = read_protocol_file(args.protocol_path)
    print(f"Found {len(file_list)} files to process")
    
    if len(file_list) == 0:
        print("❌ No files found in protocol file")
        return
    
    # Create output directory
    os.makedirs(args.base_output_dir, exist_ok=True)
    
    # Determine processing method
    if args.sequential or args.num_workers == 1:
        print("Using sequential processing")
        results, errors = process_files_sequential(
            file_list, args.base_input_dir, args.base_output_dir, 
            args.noise_type, scale_pool, args.sample_rate, args.frame_length
        )
    else:
        # Use threading instead of multiprocessing
        if args.num_workers is None:
            num_workers = 4  # Conservative default for threading
        else:
            num_workers = args.num_workers
        
        print(f"Using threaded processing with {num_workers} threads")
        
        results_container = ThreadSafeResults()
        
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                futures = []
                for file_info in file_list:
                    future = executor.submit(
                        process_single_file,
                        file_info,
                        args.base_input_dir,
                        args.base_output_dir,
                        args.noise_type,
                        scale_pool,
                        args.sample_rate,
                        args.frame_length,
                        results_container
                    )
                    futures.append(future)
                
                # Wait for completion with progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                    try:
                        future.result()  # This will raise any exceptions that occurred
                    except Exception as e:
                        print(f"❌ Task failed: {e}")
                        
        except Exception as e:
            print(f"❌ Critical error in threaded processing: {e}")
            traceback.print_exc()
            return
        
        results, errors = results_container.get_results()
    
    # Save configuration
    config = {
        'protocol_path': args.protocol_path,
        'base_input_dir': args.base_input_dir,
        'base_output_dir': args.base_output_dir,
        'noise_type': args.noise_type,
        'num_workers': args.num_workers if not args.sequential else 1,
        'sample_rate': args.sample_rate,
        'use_common_scales': args.use_common_scales,
        'frame_length': args.frame_length,
        'seed': args.seed,
        'sequential': args.sequential,
        'total_scales': len(scale_pool),
        'total_files': len(file_list)
    }
    
    config_path = os.path.join(args.base_output_dir, f"config_{args.noise_type}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    # Create metadata CSV
    if results:
        metadata_df = pd.DataFrame(results)
        metadata_path = os.path.join(args.base_output_dir, f"metadata_{args.noise_type}.csv")
        metadata_df.to_csv(metadata_path, index=False)
        print(f"✅ Metadata saved to: {metadata_path}")
    
    # Create error log
    if errors:
        error_df = pd.DataFrame(errors)
        error_path = os.path.join(args.base_output_dir, f"errors_{args.noise_type}.csv")
        error_df.to_csv(error_path, index=False)
        print(f"❌ Error log saved to: {error_path}")
    
    # Summary
    print(f"\n📊 Processing Summary:")
    print(f"Total files: {len(file_list)}")
    print(f"Successfully processed: {len(results)}")
    print(f"Errors: {len(errors)}")
    if len(file_list) > 0:
        print(f"Success rate: {len(results)/len(file_list)*100:.2f}%")
    
    # Create protocol file for processed data
    if results:
        output_protocol_path = os.path.join(args.base_output_dir, f"protocol_{args.noise_type}.txt")
        with open(output_protocol_path, 'w') as f:
            for result in results:
                f.write(f"{result['output_path']} {result['split']} {result['new_label']}\n")
        print(f"✅ Output protocol saved to: {output_protocol_path}")

if __name__ == "__main__":
    main()