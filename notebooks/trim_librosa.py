from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pyaudio
import matplotlib.pylab as plt
import matplotlib
import torchaudio
import io
import numpy as np
import torch
import random
import librosa


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# torchaudio.set_audio_backend("soundfile")
set_seed(1)
torch.set_num_threads(1)

# Define constants
DATASET_PATH = "/data/hungdx/Lightning-hydra/data/0_large-corpus"
PROTOCOL_PATH = "/data/hungdx/Lightning-hydra/data/0_large-corpus/protocol.txt"

SAMPLE_RATE = 16000
CUT_SIZE = 16000


DESTINATION_PATH = f"/data/hungdx/Lightning-hydra/data/0_large-corpus/trim_librosa"

if not os.path.exists(DESTINATION_PATH):
    os.makedirs(DESTINATION_PATH)

THRESHOLD = 0.5  # Threshold for VAD to determine speech


def process_line(line):
    line = line.strip().split()
    file = line[0]
    src_file_path = os.path.join(DATASET_PATH, file)
    waveform, sample_rate = librosa.load(src_file_path, sr=SAMPLE_RATE)

    waveform = librosa.effects.trim(waveform)[0]

    # convert to tensor
    waveform = torch.from_numpy(waveform)

    dst_file_path = os.path.join(DESTINATION_PATH, file)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

    results = []

    torchaudio.save(dst_file_path, waveform.unsqueeze(0), SAMPLE_RATE)
    results.append(dst_file_path)
    return results


# Reading the lines in advance to prevent file access issues in threads
with open(PROTOCOL_PATH) as file:
    lines = file.readlines()

executor = ProcessPoolExecutor(max_workers=40)
futures = [executor.submit(process_line, line) for line in lines]


for future in tqdm(as_completed(futures), total=len(futures)):
    result = future.result()
