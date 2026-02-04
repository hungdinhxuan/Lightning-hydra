import os
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal.windows import hamming
from scipy.fft import fft, ifft
import multiprocessing as mp
from multiprocessing import Pool, Manager
from functools import partial
import time
import librosa
from tqdm import tqdm
import argparse


# ============================================================================
# SSBoll79 Implementation (Spectral Subtraction)
# ============================================================================

def SSBoll79(signal, fs, IS=0.25):
    """
    Spectral Subtraction based on Boll 79. Amplitude spectral subtraction 
    Includes Magnitude Averaging and Residual noise Reduction
    
    Parameters:
    -----------
    signal : array-like
        The noisy signal
    fs : int
        Sampling frequency
    IS : float, optional
        Initial silence (noise only) length in seconds (default: 0.25 sec)
    
    Returns:
    --------
    output : ndarray
        Denoised signal
    """
    W = int(0.025 * fs)  # Window length is 25 ms
    nfft = W
    SP = 0.4  # Shift percentage is 40% (10ms)
    wnd = hamming(W)
    
    NIS = int((IS * fs - W) / (SP * W) + 1)  # Number of initial silence segments
    Gamma = 1  # Magnitude Power (1 for magnitude, 2 for power spectrum)
    
    y = segment(signal, W, SP, wnd)
    Y = fft(y, nfft, axis=0)
    YPhase = np.angle(Y[:int(Y.shape[0]/2)+1, :])  # Noisy Speech Phase
    Y = np.abs(Y[:int(Y.shape[0]/2)+1, :]) ** Gamma  # Spectrogram
    
    numberOfFrames = Y.shape[1]
    FreqResol = Y.shape[0]
    
    N = np.mean(Y[:, :NIS], axis=1)  # Initial Noise Power Spectrum mean
    NRM = np.zeros_like(N)  # Noise Residual Maximum
    NoiseCounter = 0
    NoiseLength = 9  # Smoothing factor for noise updating
    Beta = 0.03
    
    # Magnitude Averaging
    YS = Y.copy()
    for i in range(1, numberOfFrames - 1):
        YS[:, i] = (Y[:, i-1] + Y[:, i] + Y[:, i+1]) / 3
    
    X = np.zeros_like(Y)
    
    for i in range(numberOfFrames):
        NoiseFlag, SpeechFlag, NoiseCounter, Dist = vad(
            Y[:, i] ** (1/Gamma), 
            N ** (1/Gamma), 
            NoiseCounter
        )
        
        if SpeechFlag == 0:
            N = (NoiseLength * N + Y[:, i]) / (NoiseLength + 1)  # Update noise
            NRM = np.maximum(NRM, YS[:, i] - N)  # Update Maximum Noise Residue
            X[:, i] = Beta * Y[:, i]
        else:
            D = YS[:, i] - N  # Spectral Subtraction
            
            if i > 0 and i < numberOfFrames - 1:  # Residual Noise Reduction
                for j in range(len(D)):
                    if D[j] < NRM[j]:
                        D[j] = min([D[j], YS[j, i-1] - N[j], YS[j, i+1] - N[j]])
            
            X[:, i] = np.maximum(D, 0)
    
    output = OverlapAdd2(X ** (1/Gamma), YPhase, W, int(SP * W))
    output = output / np.max(np.abs(output))
    
    return output


def OverlapAdd2(XNEW, yphase, windowLen, ShiftLen):
    """Reconstruct signal from spectrogram using overlap-add method"""
    ShiftLen = int(ShiftLen)
    FreqRes, FrameNum = XNEW.shape
    
    Spec = XNEW * np.exp(1j * yphase)
    
    # Mirror spectrum for real signal
    if windowLen % 2:  # if windowLen is odd
        Spec = np.vstack([Spec, np.flipud(np.conj(Spec[1:, :]))])
    else:
        Spec = np.vstack([Spec, np.flipud(np.conj(Spec[1:-1, :]))])
    
    sig = np.zeros((FrameNum - 1) * ShiftLen + windowLen)
    
    for i in range(FrameNum):
        start = i * ShiftLen
        spec = Spec[:, i]
        sig[start:start + windowLen] += np.real(ifft(spec, windowLen))
    
    return sig


def vad(signal, noise, NoiseCounter, NoiseMargin=3, Hangover=8):
    """Spectral Distance Voice Activity Detector"""
    # Avoid log of zero
    signal = np.maximum(signal, 1e-10)
    noise = np.maximum(noise, 1e-10)
    
    SpectralDist = 20 * (np.log10(signal) - np.log10(noise))
    SpectralDist[SpectralDist < 0] = 0
    Dist = np.mean(SpectralDist)
    
    if Dist < NoiseMargin:
        NoiseFlag = 1
        NoiseCounter = NoiseCounter + 1
    else:
        NoiseFlag = 0
        NoiseCounter = 0
    
    # Detect noise only periods
    if NoiseCounter > Hangover:
        SpeechFlag = 0
    else:
        SpeechFlag = 1
    
    return NoiseFlag, SpeechFlag, NoiseCounter, Dist


def segment(signal, W=256, SP=0.4, Window=None):
    """Segment signal into overlapping windowed segments"""
    if Window is None:
        Window = hamming(W)
    
    Window = Window.reshape(-1, 1)  # Make it a column vector
    L = len(signal)
    SP = int(W * SP)
    N = int((L - W) / SP + 1)  # Number of segments
    
    # Create index matrix
    Index = np.tile(np.arange(W), (N, 1)) + np.tile(np.arange(N).reshape(-1, 1) * SP, (1, W))
    Index = Index.T
    
    hw = np.tile(Window, (1, N))
    Seg = signal[Index] * hw
    
    return Seg


# ============================================================================
# Preprocessing Function (Global for multiprocessing)
# ============================================================================

# Global variable to store IS parameter for multiprocessing
_IS_PARAM = 0.25

def preprocess_func(signal, fs):
    """
    Global preprocessing function for multiprocessing compatibility
    
    Parameters:
    -----------
    signal : array-like
        Input audio signal
    fs : int
        Sampling frequency
    
    Returns:
    --------
    ndarray : Processed signal
    """
    return SSBoll79(signal, fs, IS=_IS_PARAM)

def set_is_parameter(IS):
    """
    Set the IS parameter for the global preprocessing function
    
    Parameters:
    -----------
    IS : float
        Initial silence length in seconds for SSBoll79
    """
    global _IS_PARAM
    _IS_PARAM = IS


# ============================================================================
# Parallel Processing Functions
# ============================================================================

def discover_datasets(parent_dir):
    """
    Discover all dataset subdirectories within parent directory that contain protocol.txt
    
    Parameters:
    -----------
    parent_dir : str
        Path to parent directory containing multiple dataset subdirectories
    
    Returns:
    --------
    list : List of dataset directory paths
    """
    parent_path = Path(parent_dir)
    datasets = []
    
    for item in parent_path.iterdir():
        if item.is_dir():
            protocol_path = item / "protocol.txt"
            if protocol_path.exists():
                datasets.append(str(item))
    
    return datasets


def parse_protocol_file(dataset_dir, parent_dir):
    """
    Parse protocol.txt file from dataset directory to extract file paths
    Only processes 'eval' subset entries
    
    Parameters:
    -----------
    dataset_dir : str
        Path to dataset directory containing protocol.txt
    parent_dir : str
        Path to parent directory (for maintaining relative structure)
    
    Returns:
    --------
    list : List of tuples (source_path, target_relative_path, label, split)
    """
    file_entries = []
    dataset_path = Path(dataset_dir)
    parent_path = Path(parent_dir)
    protocol_path = dataset_path / "protocol.txt"
    
    if not protocol_path.exists():
        raise FileNotFoundError(f"protocol.txt not found in dataset directory: {dataset_dir}")
    
    # Get dataset name for maintaining structure
    dataset_name = dataset_path.name
    
    with open(protocol_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) >= 3:
                relative_path = parts[0]
                split = parts[1]
                label = parts[2]
                
                # Only process eval subset
                if split.lower() != 'eval':
                    continue
                
                # Construct full source path (relative to dataset directory)
                # The relative_path in protocol already includes the full path structure
                source_path = dataset_path / relative_path
                
                # Create target relative path maintaining the same structure as input
                # Include dataset name as the top-level folder to preserve hierarchy
                target_relative_path = f"{dataset_name}/{relative_path}"
                
                file_entries.append((str(source_path), str(target_relative_path), label, split))
    
    return file_entries


def process_single_audio_file(args):
    """
    Process a single audio file - designed for parallel execution
    
    Parameters:
    -----------
    args : tuple
        (source_path, target_relative_path, label, split, target_base_dir, preprocess_func, post_name)
    
    Returns:
    --------
    tuple : (success, file_info, error_message)
    """
    source_path, target_relative_path, label, split, target_base_dir, preprocess_func, post_name = args
    
    try:
        # Check if source file exists
        if not Path(source_path).exists():
            return (False, f"{target_relative_path}", f"Source file not found: {source_path}")
        
        # Read audio file
        y, Fs = sf.read(source_path)
        
        # Convert stereo to mono if needed
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        # Apply preprocessing
        output = preprocess_func(y, Fs)
        
        # Create target path maintaining the same structure as protocol
        target_path = Path(target_base_dir) / target_relative_path
        target_path = target_path.parent / (target_path.stem + post_name + target_path.suffix)
        
        # Ensure target directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write processed audio
        sf.write(target_path, output, Fs)
        
        return (True, f"{target_relative_path} -> {target_path.relative_to(Path(target_base_dir))}", None)
        
    except Exception as e:
        return (False, f"{target_relative_path}", str(e))


def process_dataset_parallel(dataset_dir, parent_dir, target_base_dir, 
                           preprocess_func, post_name, num_processes=None):
    """
    Process a single dataset using parallel processing
    
    Parameters:
    -----------
    dataset_dir : str
        Path to dataset directory containing protocol.txt and audio files
    parent_dir : str
        Path to parent directory (for maintaining structure)
    target_base_dir : str
        Base directory where processed files will be saved
    preprocess_func : callable
        Function to apply to each audio file
    post_name : str
        Suffix to add to processed files
    num_processes : int, optional
        Number of parallel processes (default: CPU count)
    
    Returns:
    --------
    tuple : (success_count, total_count, errors)
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Parse protocol file from dataset directory
    try:
        file_entries = parse_protocol_file(dataset_dir, parent_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 0, 0, []
    
    if not file_entries:
        print(f"No files found in dataset: {dataset_dir}")
        return 0, 0, []
    
    dataset_name = Path(dataset_dir).name
    print(f"Processing {len(file_entries)} files from dataset: {dataset_name}")
    
    # Prepare arguments for parallel processing
    process_args = [
        (source_path, target_relative_path, label, split, target_base_dir, preprocess_func, post_name)
        for source_path, target_relative_path, label, split in file_entries
    ]
    
    # Process files in parallel
    errors = []
    success_count = 0
    
    with Pool(processes=num_processes) as pool:
        # Use tqdm for progress tracking
        results = list(tqdm(
            pool.imap(process_single_audio_file, process_args),
            total=len(process_args),
            desc=f"Processing {dataset_name}"
        ))
    
    # Collect results
    for success, file_info, error_msg in results:
        if success:
            success_count += 1
        else:
            errors.append((file_info, error_msg))
    
    return success_count, len(file_entries), errors


def main():
    """Main function to process multiple datasets in parallel"""
    parser = argparse.ArgumentParser(description='Process audio datasets using SSBoll79 spectral subtraction')
    parser.add_argument('--parent_dir', type=str, required=True,
                       help='Parent directory containing multiple dataset subdirectories with protocol.txt files')
    parser.add_argument('--target_base_dir', type=str, required=True,
                       help='Base directory for processed output files (maintains same structure as input)')
    parser.add_argument('--post_name', type=str, default='_ssboll79',
                       help='Suffix to add to processed files (default: _ssboll79)')
    parser.add_argument('--num_processes', type=int, default=None,
                       help='Number of parallel processes (default: CPU count)')
    parser.add_argument('--IS', type=float, default=0.25,
                       help='Initial silence length in seconds for SSBoll79 (default: 0.25)')
    
    args = parser.parse_args()
    
    # Validate parent directory
    parent_path = Path(args.parent_dir)
    if not parent_path.exists():
        raise FileNotFoundError(f"Parent directory not found: {args.parent_dir}")
    
    # Discover all datasets within parent directory
    datasets = discover_datasets(args.parent_dir)
    
    if not datasets:
        raise FileNotFoundError(f"No datasets found in parent directory: {args.parent_dir}")
    
    print(f"Found {len(datasets)} datasets in parent directory: {args.parent_dir}")
    for dataset in datasets:
        print(f"  - {Path(dataset).name}")
    
    # Create target base directory
    Path(args.target_base_dir).mkdir(parents=True, exist_ok=True)
    
    # Set IS parameter for global preprocessing function
    set_is_parameter(args.IS)
    
    # Process each dataset
    total_success = 0
    total_files = 0
    all_errors = []
    
    start_time = time.time()
    
    for i, dataset_dir in enumerate(datasets):
        dataset_name = Path(dataset_dir).name
        print(f"\n{'='*60}")
        print(f"Processing Dataset {i+1}/{len(datasets)}: {dataset_name}")
        print(f"Dataset Directory: {dataset_dir}")
        print(f"{'='*60}")
        
        success_count, file_count, errors = process_dataset_parallel(
            dataset_dir=dataset_dir,
            parent_dir=args.parent_dir,
            target_base_dir=args.target_base_dir,
            preprocess_func=preprocess_func,
            post_name=args.post_name,
            num_processes=args.num_processes
        )
        
        total_success += success_count
        total_files += file_count
        all_errors.extend(errors)
        
        print(f"Dataset {i+1} ({dataset_name}) completed: {success_count}/{file_count} files processed successfully")
        if errors:
            print(f"Errors in dataset {i+1} ({dataset_name}): {len(errors)}")
    
    # Summary
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {total_success}/{total_files}")
    print(f"Success rate: {total_success/total_files*100:.2f}%")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Average time per file: {processing_time/total_files:.4f} seconds")
    
    if all_errors:
        print(f"\nErrors encountered: {len(all_errors)}")
        print("First 10 errors:")
        for i, (file_info, error_msg) in enumerate(all_errors[:10]):
            print(f"  {i+1}. {file_info}: {error_msg}")
        if len(all_errors) > 10:
            print(f"  ... and {len(all_errors) - 10} more errors")


if __name__ == "__main__":
    main()

