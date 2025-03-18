import pandas as pd
import sys
import os
import random
import numpy as np
import librosa
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import functools

os.chdir(os.path.dirname(__file__))  # Change to script's directory

sys.path.append("../")

from src.data.components.audio_augmentor.autotune import AutoTuneAugmentor
from src.data.components.audio_augmentor.base import BaseAugmentor

def audio_transform(filepath: str, aug_type: BaseAugmentor, config: dict, online: bool = False, lrs=False):
    """
    filepath: str, input audio file path
    aug_type: BaseAugmentor, augmentation type object
    config: dict, configuration dictionary
    online: bool, if True, return the augmented audio waveform, else save the augmented audio file
    """
    at = aug_type(config)
    at.load(filepath)
    at.transform()
    if online:
        audio = at.augmented_audio
        if lrs:
            return audio
        return pydub_to_librosa(audio)
    else:
        at.save()

def autotune_v1(audio_path, args, sr=16000):
    """
    Apply autotune augmentation to an audio file
    
    Parameters:
    -----------
    audio_path: str, path to the audio file
    args: Args object containing configuration
    sr: int, sample rate
    
    Returns:
    --------
    waveform: numpy array, the augmented audio waveform
    """
    aug_dir = args.aug_dir
    utt_id = os.path.basename(audio_path).split('.')[0]
    args.input_path = os.path.dirname(audio_path)
    aug_audio_path = os.path.join(aug_dir, 'autotune', utt_id + '.wav')
    args.output_path = os.path.join(aug_dir, 'autotune')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    args.out_format = 'wav'
    config = {
        "aug_type": "autotune",
        "output_path": args.output_path,
        "out_format": args.out_format,
        "noise_path": args.noise_path
    }
    
    if (args.online_aug):
        waveform = audio_transform(
            filepath=audio_path, aug_type=AutoTuneAugmentor, config=config, online=True, lrs=True)
        return waveform
    else:
        if os.path.exists(aug_audio_path):
            waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform
        else:
            audio_transform(
                filepath=audio_path, aug_type=AutoTuneAugmentor, config=config, online=False, lrs=True)
            waveform, _ = librosa.load(aug_audio_path, sr=sr, mono=True)
            return waveform

class Args:
    def __init__(self):
        self.aug_dir = "/nvme1/hungdx/Lightning-hydra/data/0_large-corpus/aug"
        self.online_aug = False
        self.noise_path = "/tmp/noise.wav"

def process_sample(audio_path, args):
    """
    Process a single audio sample for parallel execution
    
    Parameters:
    -----------
    audio_path: str, path to the audio file
    args: Args object containing configuration
    
    Returns:
    --------
    audio_path: str, path to the processed audio file (for tracking purposes)
    """
    try:
        # Fixed: Correctly call autotune_v1 with audio_path as the first argument
        autotune_v1(audio_path, args)
        return audio_path
    except Exception as e:
        return f"Error processing {audio_path}: {str(e)}"

def main():
    # Load dataset
    BASE_DIR = "/nvme1/hungdx/Lightning-hydra/data/0_large-corpus"
    df = pd.read_csv("/nvme1/hungdx/Lightning-hydra/notebooks/new_protocol_trim_vocoded_cleaned_v2.txt", 
                     sep=" ", header=None)
    df.columns = ["utt", "subset", "label"]
    
    # Filter to get samples with subset is train
    train_df = df[df["subset"] == "train"]
    
    # Get full paths for all training samples
    audio_paths = [os.path.join(BASE_DIR, utt) for utt in train_df["utt"].values]
    
    # Initialize args
    args = Args()
    
    # Set the maximum number of workers (adjust based on your CPU capabilities)
    max_workers = 20
    
    # Create a partial function with fixed args
    process_fn = functools.partial(process_sample, args=args)
    
    # Process all files in parallel with progress bar
    print(f"Processing {len(audio_paths)} audio files with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and track with tqdm
        results = list(tqdm(
            executor.map(process_fn, audio_paths),
            total=len(audio_paths),
            desc="Augmenting audio",
            unit="file"
        ))
    
    # Count successful and failed operations
    successful = sum(1 for r in results if not isinstance(r, str) or not r.startswith("Error"))
    failed = len(results) - successful
    
    print(f"Augmentation complete: {successful} files processed successfully, {failed} files failed")
    
    # If there were errors, print the first few
    if failed > 0:
        errors = [r for r in results if isinstance(r, str) and r.startswith("Error")]
        print(f"First {min(5, len(errors))} errors:")
        for i, error in enumerate(errors[:5]):
            print(f"{i+1}. {error}")

if __name__ == "__main__":
    main()