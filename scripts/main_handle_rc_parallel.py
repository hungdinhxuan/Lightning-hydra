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
# Parallel Processing Functions
# ============================================================================

def process_single_audio_file(args):
    """
    Process a single audio file - designed for parallel execution
    
    Parameters:
    -----------
    args : tuple
        (audio_file_path, target_dir, preprocess_func, post_name)
    
    Returns:
    --------
    tuple : (success, audio_file_name, error_message)
    """
    audio_file_path, target_dir, preprocess_func, post_name = args
    
    try:
        # Read audio file using librosa
        y, Fs = librosa.load(audio_file_path, sr=16000)
        y = librosa.to_mono(y)
        
        # Apply preprocessing
        output = preprocess_func(y, Fs)
        
        # Create output filename
        audio_file = Path(audio_file_path)
        output_filename = audio_file.stem + post_name + ".wav"
        output_path = target_dir / output_filename
        
        # Ensure target directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write processed audio
        sf.write(output_path, output, Fs)
        
        return (True, audio_file.name, None)
        
    except Exception as e:
        return (False, Path(audio_file_path).name, str(e))


def process_category_parallel(source_directory, new_folder, category_path, preprocess_func, 
                            post_name, progress_offset, total_categories, num_processes=None):
    """Process all files in a category using parallel processing"""
    category_source = Path(source_directory) / category_path
    category_target = new_folder / category_path
    
    if not category_source.exists():
        print(f"Warning: {category_source} does not exist")
        return
    
    # Get list of folders (speakers/sessions)
    folders = [f for f in category_source.iterdir() if f.is_dir()]
    M = len(folders)
    
    print(f"\nProcessing {category_path} with {M} folders...")
    
    # Collect all audio files from all folders
    all_audio_files = []
    for folder in folders:
        source_dir = category_source / folder.name
        target_dir = category_target / folder.name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        audio_files = list(source_dir.glob("*.wav"))
        for audio_file in audio_files:
            all_audio_files.append((str(audio_file), target_dir, preprocess_func, post_name))
    
    if not all_audio_files:
        print(f"No audio files found in {category_path}")
        return
    
    # Determine number of processes
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(all_audio_files))
    
    print(f"Processing {len(all_audio_files)} files using {num_processes} processes...")
    
    # Process all files in parallel
    with Pool(processes=num_processes) as pool:
        results = []
        
        # Use imap for better progress tracking
        for result in tqdm(pool.imap(process_single_audio_file, all_audio_files), 
                          total=len(all_audio_files), 
                          desc=f"Processing {category_path}"):
            results.append(result)
    
    # Report results
    successful = sum(1 for success, _, _ in results if success)
    failed = len(results) - successful
    
    print(f"Completed: {successful} successful, {failed} failed")
    
    # Print failed files
    for success, filename, error in results:
        if not success:
            print(f"Failed to process {filename}: {error}")
    
    return successful


def create_protocol_file(new_folder):
    """Create protocol.txt file listing all processed audio files"""
    protocol_path = new_folder / "protocol.txt"
    
    categories = [
        ("benign/en", "bonafide"),
        ("spoof/bark/en", "spoof"),
        ("spoof/vits/en", "spoof"),
        ("spoof/xtts_v1.1/en", "spoof"),
        ("spoof/xtts_v2/en", "spoof")
    ]
    
    with open(protocol_path, 'w') as f:
        for category_path, label in categories:
            category_dir = new_folder / category_path
            
            if not category_dir.exists():
                continue
            
            # Get all speaker/session folders
            folders = sorted([d for d in category_dir.iterdir() if d.is_dir()])
            
            for folder in folders:
                # Get all audio files in this folder
                audio_files = sorted(folder.glob("*.wav"))
                
                for audio_file in audio_files:
                    # Create relative path for protocol
                    relative_path = audio_file.relative_to(new_folder)
                    # Convert Windows backslashes to forward slashes
                    relative_path_str = str(relative_path).replace('\\', '/')
                    
                    # Write protocol line
                    f.write(f"{relative_path_str} eval {label}\n")
    
    print(f"Protocol file created at: {protocol_path}")


def setup_directories(root_directory, source_directory, name_processing):
    """Set up directory structure for processed files"""
    new_folder = Path(root_directory) / name_processing / "test"
    new_folder.mkdir(parents=True, exist_ok=True)
    
    # Copy directory structure (equivalent to xcopy /T)
    source_path = Path(source_directory)
    for dirpath, dirnames, filenames in os.walk(source_path):
        relative_path = Path(dirpath).relative_to(source_path)
        target_dir = new_folder / relative_path
        target_dir.mkdir(parents=True, exist_ok=True)
    
    return new_folder


def main():
    """Main preprocessing pipeline with parallel processing support"""
    parser = argparse.ArgumentParser(description='Parallel Audio Preprocessing Pipeline')
    parser.add_argument('--root-dir', type=str, 
                       default="/nvme1/hungdx/Lightning-hydra/data/SSBoll_resample_librosa",
                       help='Root directory for output')
    parser.add_argument('--source-dir', type=str,
                       default="/nvme1/hungdx/Lightning-hydra/data/wildspoof_challenge_benchmark/record",
                       help='Source directory containing audio files')
    parser.add_argument('--name-processing', type=str, default="ssBoll_py_parallel",
                       help='Name for processing output folder')
    parser.add_argument('--post-name', type=str, default="_ssBoll_py",
                       help='Suffix for processed files')
    parser.add_argument('--num-processes', type=int, default=8,
                       help='Number of parallel processes (default: CPU count)')
    parser.add_argument('--categories', nargs='+', 
                       default=["benign/en", "spoof/bark/en", "spoof/vits/en", 
                               "spoof/xtts_v1.1/en", "spoof/xtts_v2/en"],
                       help='Categories to process')
    
    args = parser.parse_args()
    
    # Configuration
    root_directory = args.root_dir
    source_directory = args.source_dir
    name_processing = args.name_processing
    post_name = args.post_name
    num_processes = args.num_processes
    categories = args.categories
    
    print(f"Starting parallel audio preprocessing...")
    print(f"Root directory: {root_directory}")
    print(f"Source directory: {source_directory}")
    print(f"Output folder: {name_processing}")
    print(f"Number of processes: {num_processes or mp.cpu_count()}")
    print(f"Categories: {categories}")
    
    start_time = time.time()
    
    # Setup directories
    print("\nSetting up directories...")
    new_folder = setup_directories(root_directory, source_directory, name_processing)
    print(f"Output directory: {new_folder}")
    
    # Process each category
    print("\nProcessing audio files...")
    total_processed = 0
    for category_path in categories:
        print(f"\nProcessing {category_path}...")
        processed_count = process_category_parallel(
            source_directory, 
            new_folder, 
            category_path, 
            SSBoll79,  # Using the spectral subtraction function
            post_name, 
            0,  # Progress offset not used in parallel version
            len(categories),
            num_processes
        )
        if processed_count is not None:
            total_processed += processed_count
    
    # Create protocol file
    print("\nCreating protocol file...")
    create_protocol_file(new_folder)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nProcessing complete!")
    print(f"Total files processed: {total_processed}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per category: {total_time/len(categories):.2f} seconds")
    if total_processed > 0:
        print(f"Average time per file: {total_time/total_processed:.2f} seconds")


if __name__ == "__main__":
    main()
