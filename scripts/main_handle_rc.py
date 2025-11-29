import os
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal.windows import hamming
from scipy.fft import fft, ifft


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
# Audio Processing Pipeline
# ============================================================================

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


def process_audio_files(source_dir, target_dir, preprocess_func, post_name="_ssBoll_py"):
    """Process audio files in a directory"""
    audio_files = list(source_dir.glob("*.wav"))
    
    for audio_file in audio_files:
        try:
            # Read audio file
            y, Fs = sf.read(audio_file)
            
            # Convert stereo to mono if needed
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            
            # Apply preprocessing
            output = preprocess_func(y, Fs)
            
            # Create output filename
            output_filename = audio_file.stem + post_name + ".wav"
            output_path = target_dir / output_filename
            
            # Write processed audio
            sf.write(output_path, output, Fs)
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")


def process_category(source_directory, new_folder, category_path, preprocess_func, 
                     post_name, progress_offset, total_categories):
    """Process all files in a category"""
    category_source = Path(source_directory) / category_path
    category_target = new_folder / category_path
    
    if not category_source.exists():
        print(f"Warning: {category_source} does not exist")
        return
    
    # Get list of folders (speakers/sessions)
    folders = [f for f in category_source.iterdir() if f.is_dir()]
    M = len(folders)
    
    for ct1, folder in enumerate(folders, 1):
        source_dir = category_source / folder.name
        target_dir = category_target / folder.name
        
        # Process all audio files in this folder
        process_audio_files(source_dir, target_dir, preprocess_func, post_name)
        
        # Progress update
        progress = (ct1 / M * 100 / total_categories) + progress_offset
        print(f"{progress:.2f}%")


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


def main():
    """Main preprocessing pipeline"""
    # Configuration
    root_directory = r"F:\My_cloud\Deepfake\MATLAB_code\modified_dataset_replayDF"
    source_directory = r"F:\My_cloud\Deepfake\MATLAB_code\modified_dataset_replayDF\recording\test"
    name_processing = "ssBoll_py"
    post_name = "_ssBoll_py"
    
    # Setup directories
    print("Setting up directories...")
    new_folder = setup_directories(root_directory, source_directory, name_processing)
    print(f"Output directory: {new_folder}")
    
    print("0.00%")
    
    # Define categories to process
    categories = [
        ("benign/en", 0),
        ("spoof/bark/en", 20),
        ("spoof/vits/en", 40),
        ("spoof/xtts_v1.1/en", 60),
        ("spoof/xtts_v2/en", 80)
    ]
    
    total_categories = len(categories)
    src/main.py
    # Process each category
    print("\nProcessing audio files...")
    for category_path, progress_offset in categories:
        print(f"\nProcessing {category_path}...")
        process_category(
            source_directory, 
            new_folder, 
            category_path, 
            SSBoll79,  # Using the spectral subtraction function
            post_name, 
            progress_offset, 
            total_categories
        )
    
    # Create protocol file
    print("\nCreating protocol file...")
    create_protocol_file(new_folder)
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()