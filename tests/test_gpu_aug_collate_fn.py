import os
import sys
import numpy as np
import torch
import tempfile
import shutil
import glob
import soundfile as sf

# Ensure imports work in pytest (some modules expect `core_scripts` to be importable)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def test_apply_gpu_augmentation_to_batch_low_pass_filter_shape_dtype():
    """Unit test for apply_gpu_augmentation_to_batch: shape + dtype stability."""
    from src.data.components.collate_fn import apply_gpu_augmentation_to_batch

    batch_size = 4
    n_samples = 16000
    x = torch.randn(batch_size, n_samples, dtype=torch.float32)

    args = {
        "min_cutoff_freq": 2000,
        "max_cutoff_freq": 7500,
        "p": 1.0,
    }

    y = apply_gpu_augmentation_to_batch(
        x,
        augmentation_methods=["low_pass_filter"],
        args=args,
        sample_rate=16000,
        device=None,
    )

    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape
    assert y.dtype == torch.float32


def test_apply_gpu_augmentation_to_batch_no_methods_returns_input():
    """If no augmentation methods are provided, output must equal input."""
    from src.data.components.collate_fn import apply_gpu_augmentation_to_batch

    x = torch.randn(2, 8000, dtype=torch.float32)
    y = apply_gpu_augmentation_to_batch(
        x,
        augmentation_methods=[],
        args={},
        sample_rate=16000,
        device=None,
    )

    assert torch.allclose(x, y)


class _DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n_items: int = 8, n_samples: int = 8000):
        self.n_items = n_items
        self.n_samples = n_samples

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        # return numpy audio + integer label
        x = np.random.randn(self.n_samples).astype(np.float32)
        y = int(idx % 2)
        return x, y


def test_multi_view_collate_fn_with_gpu_augmentation_dataloader_worker_cpu():
    """
    Integration test: collate_fn must work under DataLoader worker process (forked).

    This is specifically to catch the common failure:
    RuntimeError: Cannot re-initialize CUDA in forked subprocess
    """
    from src.data.components.collate_fn import multi_view_collate_fn_with_gpu_augmentation

    dataset = _DummyDataset(n_items=6, n_samples=4000)

    # 1-second view @ 16k would require padding; here we keep sample_rate small for speed.
    sample_rate = 1000
    views = [1, 2]
    view_padding_configs = {"1": {"padding_type": "repeat", "random_start": False},
                            "2": {"padding_type": "repeat", "random_start": False}}

    args = {
        "min_cutoff_freq": 200,
        "max_cutoff_freq": 400,
        "p": 1.0,
    }

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3,
        num_workers=1,  # important: run collate in worker process
        shuffle=False,
        collate_fn=lambda batch: multi_view_collate_fn_with_gpu_augmentation(
            batch,
            views=views,
            sample_rate=sample_rate,
            padding_type="repeat",
            random_start=False,
            view_padding_configs=view_padding_configs,
            augmentation_methods=["low_pass_filter"],
            args=args,
            device=None,
        ),
        persistent_workers=False,
    )

    out = next(iter(loader))
    assert isinstance(out, dict)
    for v in views:
        x_batch, y_batch = out[v]
        assert isinstance(x_batch, torch.Tensor)
        assert isinstance(y_batch, torch.Tensor)
        assert x_batch.shape[0] == 3
        assert y_batch.shape[0] == 3
        # expected padded length = view * sample_rate
        assert x_batch.shape[1] == v * sample_rate


def test_multi_view_collate_fn_with_gpu_augmentation_no_methods_passthrough():
    """If no augmentation methods are given, collate_fn_with_gpu_augmentation should behave like plain multi_view_collate_fn."""
    from src.data.components.collate_fn import (
        multi_view_collate_fn,
        multi_view_collate_fn_with_gpu_augmentation,
    )

    dataset = _DummyDataset(n_items=4, n_samples=3000)
    sample_rate = 1000
    views = [1]
    view_padding_configs = {"1": {"padding_type": "repeat", "random_start": False}}

    # Build a small batch manually from dataset
    raw_batch = [dataset[i] for i in range(2)]

    plain = multi_view_collate_fn(
        raw_batch,
        views=views,
        sample_rate=sample_rate,
        padding_type="repeat",
        random_start=False,
        view_padding_configs=view_padding_configs,
    )

    wrapped = multi_view_collate_fn_with_gpu_augmentation(
        raw_batch,
        views=views,
        sample_rate=sample_rate,
        padding_type="repeat",
        random_start=False,
        view_padding_configs=view_padding_configs,
        augmentation_methods=[],  # no methods
        args={},
        device=None,
    )

    assert plain.keys() == wrapped.keys()
    for v in views:
        x_plain, y_plain = plain[v]
        x_wrap, y_wrap = wrapped[v]
        assert torch.allclose(x_plain, x_wrap)
        assert torch.allclose(y_plain, y_wrap)


class _RealAudioDataset(torch.utils.data.Dataset):
    """Dataset that loads real audio files from a folder."""
    def __init__(self, audio_folder: str, sample_rate: int = 16000):
        self.audio_folder = audio_folder
        self.sample_rate = sample_rate
        # Find all audio files in folder
        self.audio_files = []
        for ext in ['*.wav', '*.flac', '*.mp3']:
            self.audio_files.extend(glob.glob(os.path.join(audio_folder, ext)))
        self.audio_files.sort()
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_folder}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        import librosa
        audio_path = self.audio_files[idx]
        # Load audio using librosa (same as load_audio function)
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        # Return numpy array + dummy label
        label = int(idx % 2)
        return audio, label


def test_gpu_augmentation_with_real_audio_files_save_output():
    """
    Integration test: Load real audio files from folder, apply GPU augmentation,
    and save augmented audio to output folder.
    
    This test:
    1. Creates temporary input folder with synthetic audio files
    2. Loads audio files through dataset
    3. Applies augmentation via collate_fn
    4. Saves augmented audio to output folder
    5. Verifies output files exist and can be loaded
    """
    from src.data.components.collate_fn import multi_view_collate_fn_with_gpu_augmentation
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        input_folder = os.path.join(temp_dir, "input_audio")
        output_folder = os.path.join(temp_dir, "output_augmented")
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        
        # Create synthetic audio files (simulating real audio)
        sample_rate = 16000
        n_files = 3
        audio_files = []
        
        for i in range(n_files):
            # Generate synthetic audio (sine wave + noise)
            duration = 1.0  # 1 second
            t = np.linspace(0, duration, int(sample_rate * duration))
            freq = 440 + i * 100  # Different frequency for each file
            audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
            # Add some noise
            audio += 0.1 * np.random.randn(len(audio)).astype(np.float32)
            # Normalize
            audio = audio / (np.abs(audio).max() + 1e-8)
            
            # Save audio file
            audio_path = os.path.join(input_folder, f"test_audio_{i:02d}.wav")
            sf.write(audio_path, audio, sample_rate)
            audio_files.append(audio_path)
        
        # Create dataset from input folder
        dataset = _RealAudioDataset(input_folder, sample_rate=sample_rate)
        assert len(dataset) == n_files
        
        # Setup augmentation config
        views = [1]  # Single view for simplicity
        view_padding_configs = {"1": {"padding_type": "repeat", "random_start": False}}
        
        args = {
            "min_cutoff_freq": 2000,
            "max_cutoff_freq": 7500,
            "p": 1.0,
        }
        
        # Create DataLoader with GPU augmentation collate_fn
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            num_workers=0,  # Use 0 to avoid CUDA issues in test
            shuffle=False,
            collate_fn=lambda batch: multi_view_collate_fn_with_gpu_augmentation(
                batch,
                views=views,
                sample_rate=sample_rate,
                padding_type="repeat",
                random_start=False,
                view_padding_configs=view_padding_configs,
                augmentation_methods=["low_pass_filter"],
                args=args,
                device=None,
            ),
        )
        
        # Process batches and save augmented audio
        output_files = []
        for batch_idx, batch_output in enumerate(loader):
            assert isinstance(batch_output, dict)
            
            # Get augmented audio for view 1
            augmented_batch, labels_batch = batch_output[1]
            assert isinstance(augmented_batch, torch.Tensor)
            assert augmented_batch.ndim == 2  # (batch_size, samples)
            
            # Save each augmented audio in batch
            for i in range(augmented_batch.shape[0]):
                # Convert tensor to numpy
                augmented_audio = augmented_batch[i].cpu().numpy()
                
                # Create output filename
                input_filename = os.path.basename(audio_files[batch_idx * 2 + i])
                output_filename = f"augmented_{input_filename}"
                output_path = os.path.join(output_folder, output_filename)
                
                # Save augmented audio
                sf.write(output_path, augmented_audio, sample_rate)
                output_files.append(output_path)
        
        # Verify output files exist
        assert len(output_files) == n_files
        for output_file in output_files:
            assert os.path.exists(output_file), f"Output file {output_file} does not exist"
            
            # Verify file can be loaded
            loaded_audio, loaded_sr = sf.read(output_file)
            assert loaded_sr == sample_rate
            assert len(loaded_audio) > 0
            assert isinstance(loaded_audio, np.ndarray)
        
        # Verify augmented audio is different from original (at least for some files)
        # Load original and compare
        for i, (original_path, augmented_path) in enumerate(zip(audio_files, output_files)):
            original_audio, _ = sf.read(original_path)
            augmented_audio, _ = sf.read(augmented_path)
            
            # They should have similar length (may differ due to padding)
            assert abs(len(original_audio) - len(augmented_audio)) <= sample_rate, \
                f"Length mismatch too large: {len(original_audio)} vs {len(augmented_audio)}"
            
            # Augmented audio should be different (low-pass filter changes the signal)
            # Check that they're not identical
            if len(original_audio) == len(augmented_audio):
                # Normalize for comparison
                orig_norm = original_audio / (np.abs(original_audio).max() + 1e-8)
                aug_norm = augmented_audio / (np.abs(augmented_audio).max() + 1e-8)
                
                # They should not be identical (low-pass filter modifies frequency content)
                max_diff = np.abs(orig_norm - aug_norm).max()
                assert max_diff > 1e-6, \
                    f"Augmented audio is too similar to original (max_diff={max_diff})"
        
        print(f"✓ Successfully processed {n_files} audio files")
        print(f"✓ Augmented audio saved to: {output_folder}")
        print(f"✓ Output files: {[os.path.basename(f) for f in output_files]}")

