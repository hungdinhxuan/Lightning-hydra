#!/usr/bin/env python
"""script to create protocol for AIShell3 databse
No additional protocol used, we simply walk through the directory,
and no fake audios in this database

/path/to/your/AIShell3/
├── test/
│   ├── wav/
│   │   ├── SSB0005/
│   │   │   ├── xx.wav
│   │   ├── . . . 
├── train/
│   ├── . . . 

AIShell3.csv:
"""
import os
import sys
import csv
import glob

try:
    import pandas as pd
    from pandarallel import pandarallel
    import torchaudio
except ImportError:
    print("Please install pandas, pandarallel and torchaudio")
    sys.exit(1)


__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

# used for pandas pd.parallel_apply() to speed up
pandarallel.initialize()

# Define paths 
root_folder = '/home/hungdx/code/Lightning-hydra/data/Post_training_benchmark/'
dataset_name = 'AISHELL-3'
data_folder = os.path.join(root_folder, dataset_name)
ID_PREFIX = 'AISHELL-3-'
output_csv = dataset_name + '.csv'

# Function to collect file paths and basic metadata
def collect_file_paths(data_folder):
    file_data = []
    # List all wav files
    for file_path in sorted(
        glob.glob(os.path.join(data_folder, "**", "*.wav"), recursive=True)
    ):
        relative_path = file_path.replace(root_folder, "$ROOT/")
        # Extract relevant folder names
        parts = os.path.normpath(relative_path).split(os.sep)
        # ['$ROOT', 'AIShell3', 'train', 'wav', 'SSB1567', 'SSB15670385.wav']
        speaker = parts[4]
        subset = parts[2]
        if 'train' in subset:
            proportion = 'train'
        elif 'test' in subset:
            proportion = 'test'
        attack = '-'
        label = 'bonafide'
        language = 'ZH'
        file_id = f"{speaker}-{os.path.splitext(parts[-1])[0]}"
        
        file_data.append({
            "file_path": file_path,
            "relative_path": relative_path,
            "proportion": proportion,
            "label": label,
            "attack": attack,
            "speaker": speaker,
            "file_id": file_id,
            "language": language
        })
    return file_data

# Function to process audio metadata in parallel
def collect_audio_metadata(file_data):
    def __get_audio_meta(row):
        file_path = row["file_path"]
        if os.path.exists(file_path):
            try:
                # Load metainfo with torchaudio
                metainfo = torchaudio.info(file_path)
                # Return metadata
                return {
                    "ID": ID_PREFIX + row["file_id"],
                    "Label": row["label"],
                    "SampleRate": metainfo.sample_rate,
                    "Duration": round(metainfo.num_frames / metainfo.sample_rate, 2),
                    "Path": row["relative_path"],
                    "Attack": row["attack"],
                    "Speaker": row["speaker"],
                    "Proportion": row["proportion"],
                    "AudioChannel": metainfo.num_channels,
                    "AudioEncoding": metainfo.encoding,
                    "AudioBitSample": metainfo.bits_per_sample,
                    "Language": row["language"],
                }
            except Exception as e:
                # Handle any exception and skip this file
                print(f"Error: Could not load file {file_path}. Skipping. Reason: {e}")
                return None
        else:
            print(f"Warning: File {file_path} does not exist, skipping entry.")
            return None
    
    # Convert to DataFrame for parallel processing
    df = pd.DataFrame(file_data)
    # Use parallel_apply for speedup
    metadata = df.parallel_apply(lambda x: __get_audio_meta(x), axis=1)
    # Filter out None values (failed files)
    metadata = metadata.dropna().tolist()
    return metadata

# Write metadata to CSV
def write_csv(metadata):
    header = ["ID", "Label", "Duration", "SampleRate", "Path", "Attack", "Speaker",\
              "Proportion", "AudioChannel", "AudioEncoding", "AudioBitSample",\
              "Language"]
    metadata = pd.DataFrame(metadata)
    metadata = metadata[header]
    metadata.to_csv(output_csv, index=False)
    
    metadata['Path'] = metadata['Path'].apply(lambda x: x.replace(f"$ROOT/{dataset_name}/", ""))
    metadata['subset'] = 'eval'
    # keep only two columns: Path and subset
    metadata = metadata[['Path', 'subset', 'Label']]
    output_protocol_txt = os.path.join(root_folder, dataset_name, 'protocol.txt')
    metadata.to_csv(output_protocol_txt, index=False, header=False, sep=' ')

# Main script
if __name__ == "__main__":
    # Step 1: Collect file paths and basic metadata
    print("Collecting file paths...")
    file_data = collect_file_paths(data_folder)
    print(f"Found {len(file_data)} audio files")
    
    # Step 2: Process audio metadata in parallel
    print("Processing audio metadata in parallel...")
    metadata = collect_audio_metadata(file_data)
    print(f"Successfully processed {len(metadata)} files")
    
    # Step 3: Write metadata to CSV
    write_csv(metadata)
    print(f"Metadata CSV written to {output_csv}")
