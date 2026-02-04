import noisereduce as nr
import soundfile as sf
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def stationary_noisereduction(signal, sr):
    return nr.reduce_noise(y = signal, sr=sr, n_std_thresh_stationary=1.5,stationary=True)

def non_stationary_noisereduction(signal, sr):
    return nr.reduce_noise(y = signal, sr=sr, thresh_n_mult_nonstationary=2,stationary=False)

def save_audio(signal, sr, filename):
    sf.write(filename, signal, sr)

def load_audio(filename):
    return sf.read(filename)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def process_one(absolute_path):
    signal, sr = load_audio(absolute_path)
    base = os.path.basename(absolute_path)
    return signal, sr, base

def process_and_save(absolute_path, stationary_dir, non_stationary_dir):
    signal, sr, base = process_one(absolute_path)

    # Stationary
    reduced_stationary = stationary_noisereduction(signal, sr)
    out_stationary = os.path.join(stationary_dir, base)
    save_audio(reduced_stationary, sr, out_stationary)

    # Non-stationary
    reduced_non_stationary = non_stationary_noisereduction(signal, sr)
    out_non_stationary = os.path.join(non_stationary_dir, base)
    save_audio(reduced_non_stationary, sr, out_non_stationary)

    # Return filenames (relative paths are just basenames here)
    return base

df = pd.read_csv("/nvme1/hungdx/Lightning-hydra/data/wildspoof_challenge_benchmark/asv19/protocol.txt", sep=" ", header=None)
df.columns = ["utt", "subset", "label"]
DATA_DIR = "/nvme1/hungdx/Lightning-hydra/data/wildspoof_challenge_benchmark/asv19"

DENOISE_DIR = "/data/Datasets/asv19_denoiser"
STATIONARY_DIR = os.path.join(DENOISE_DIR, "stationary")
NON_STATIONARY_DIR = os.path.join(DENOISE_DIR, "non_stationary")
ensure_dir(STATIONARY_DIR)
ensure_dir(NON_STATIONARY_DIR)

df['utt'] = df['utt'].apply(lambda x: os.path.join(DATA_DIR, x))
df_eval = df[df['subset'] == "eval"].reset_index(drop=True)

# Parallel processing
bases = []
paths = df_eval['utt'].tolist()
max_workers = 16
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_and_save, p, STATIONARY_DIR, NON_STATIONARY_DIR): idx for idx, p in enumerate(paths)}
    for i, future in enumerate(as_completed(futures), 1):
        base = future.result()
        bases.append(base)
        print(f"Processed {i} of {len(paths)}: {base}")

# Build and save protocol files with relative paths (basenames)
df_eval_out = df_eval.copy()
df_eval_out['utt'] = bases  # basenames relative to output folders

# Keep original ordering aligned with processed list; as_completed changed order, so rebuild using paths order
base_by_path = {os.path.basename(p): True for p in bases}
df_eval_out = df_eval.assign(utt=[os.path.basename(p) for p in paths])

stationary_protocol = os.path.join(STATIONARY_DIR, "protocol.txt")
non_stationary_protocol = os.path.join(NON_STATIONARY_DIR, "protocol.txt")

df_eval_out.to_csv(stationary_protocol, sep=" ", header=False, index=False)
df_eval_out.to_csv(non_stationary_protocol, sep=" ", header=False, index=False)

print("Done")