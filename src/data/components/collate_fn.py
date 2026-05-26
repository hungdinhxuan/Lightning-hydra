import torch
from src.data.components.dataio import pad, pad_tensor
import numpy as np
from torch import Tensor
from typing import List, Dict, Tuple, Union, Optional, Any
from src.core_scripts.data_io import wav_augmentation as nii_wav_aug
import random


def multi_view_collate_fn(batch, views=[1, 2, 3, 4], sample_rate=16000, padding_type='repeat', random_start=False, view_padding_configs: Dict[str, Dict[str, bool]] = None):
    '''
    Collate function to pad each sample in a batch to multiple views
    :param batch: list of tuples (x, label)
    :param views: list of views to pad each sample to
    :param sample_rate: sample rate of the audio
    :param padding_type: padding type to use
    :param random_start: whether to randomly start the sample
    :return: dictionary with keys as views and values as tuples of padded sequences and labels

    Example:
    batch = [([1, 2, 3], 0), ([1, 2, 3, 4], 1)]
    multi_view_collate_fn(batch, views=[1, 2], sample_rate=16000)
    Output:
    {
        1: (tensor([[1, 2, 3], [1, 2, 3, 4]]), tensor([0, 1])),
        2: (tensor([[1, 2, 3, 0], [1, 2, 3, 4]]), tensor([0, 1]))
    }
    '''
    # Set default configurations if none provided
    if view_padding_configs is None:
        view_padding_configs = {
            str(i): {'padding_type': 'repeat', 'random_start': False}
            for i in range(1, 5)
        }

    # Extract views from config and convert to integers
    views = [int(view) for view in view_padding_configs]

    view_batches = {view: [] for view in views}
    # Warning: padding_type and random_start are not used in this function
    # print("Warning: padding_type and random_start are not used in this function. Please use view_padding_configs instead")

    # Process each sample in the batch
    has_source = len(batch[0]) == 3 if batch else False
    for sample in batch:
        if has_source:
            x, label, source = sample
        else:
            x, label = sample
            source = None
        # Pad each sample for each view
        for view in views:
            view_length = view * sample_rate
            x_view = pad(x, padding_type=view_padding_configs[str(view)]['padding_type'],
                         max_len=view_length, random_start=view_padding_configs[str(view)]['random_start'])
            # Check if x_view is Tensor or numpy array and convert to Tensor if necessary
            if not torch.is_tensor(x_view):

                x_view = torch.from_numpy(x_view)
            if has_source:
                view_batches[view].append((x_view, label, source))
            else:
                view_batches[view].append((x_view, label))

    # Convert lists to tensors
    for view in views:
        if has_source:
            sequences, labels, sources = zip(*view_batches[view])
            padded_sequences = torch.stack(sequences)
            labels = torch.tensor(labels, dtype=torch.long)
            sources = torch.tensor(sources, dtype=torch.long)
            view_batches[view] = (padded_sequences, labels, sources)
        else:
            sequences, labels = zip(*view_batches[view])
            padded_sequences = torch.stack(sequences)
            labels = torch.tensor(labels, dtype=torch.long)
            view_batches[view] = (padded_sequences, labels)

    return view_batches


def multi_view_aux_collate_fn(batch, views=[1, 2, 3, 4], sample_rate=16000, padding_type='repeat', random_start=False, view_padding_configs: Dict[str, Dict[str, bool]] = None):
    '''
    Collate function to pad each sample in a batch to multiple views
    :param batch: list of tuples (x, label)
    :param views: list of views to pad each sample to
    :param sample_rate: sample rate of the audio
    :param padding_type: padding type to use
    :param random_start: whether to randomly start the sample
    :return: dictionary with keys as views and values as tuples of padded sequences and labels

    Example:
    batch = [([1, 2, 3], 0), ([1, 2, 3, 4], 1)]
    multi_view_collate_fn(batch, views=[1, 2], sample_rate=16000)
    Output:
    {
        1: (tensor([[1, 2, 3], [1, 2, 3, 4]]), tensor([0, 1])),
        2: (tensor([[1, 2, 3, 0], [1, 2, 3, 4]]), tensor([0, 1]))
    }
    '''
    # Set default configurations if none provided
    if view_padding_configs is None:
        view_padding_configs = {
            str(i): {'padding_type': 'repeat', 'random_start': False}
            for i in range(1, 5)
        }

    # Extract views from config and convert to integers
    views = [int(view) for view in view_padding_configs]

    view_batches = {view: [] for view in views}
    # Warning: padding_type and random_start are not used in this function
    # print("Warning: padding_type and random_start are not used in this function. Please use view_padding_configs instead")

    # Process each sample in the batch
    for x, label, aux_label in batch:
        # Pad each sample for each view
        for view in views:
            view_length = view * sample_rate
            x_view = pad(x, padding_type=view_padding_configs[str(view)]['padding_type'],
                         max_len=view_length, random_start=view_padding_configs[str(view)]['random_start'])
            # Check if x_view is Tensor or numpy array and convert to Tensor if necessary
            if not torch.is_tensor(x_view):

                x_view = torch.from_numpy(x_view)
            view_batches[view].append((x_view, label, aux_label))

    # Convert lists to tensors
    for view in views:
        sequences, labels, aux_labels = zip(*view_batches[view])
        padded_sequences = torch.stack(sequences)
        labels = torch.tensor(labels, dtype=torch.long)
        aux_labels = torch.tensor(aux_labels, dtype=torch.long)
        view_batches[view] = (padded_sequences, labels, aux_labels)

    return view_batches

def multi_view_collate_fn_for_scl(batch, views=[1, 2, 3, 4], sample_rate=16000,
                                  padding_type='repeat', random_start=False,
                                  view_padding_configs: Dict[str, Dict[str, bool]] = None):
    '''
    Collate function to pad each sample in a batch to multiple views
    :param batch: list of tuples (x, label)
    :param views: list of views to pad each sample to
    :param sample_rate: sample rate of the audio
    :param padding_type: padding type to use
    :param random_start: whether to randomly start the sample
    :return: dictionary with keys as views and values as tuples of padded sequences and labels
    '''
    # Set default configurations if none provided
    if view_padding_configs is None:
        view_padding_configs = {
            str(i): {'padding_type': 'repeat', 'random_start': False}
            for i in range(1, 5)
        }

    # Extract views from config and convert to integers
    views = [int(view) for view in view_padding_configs]

    view_batches = {view: [] for view in views}

    # Process each sample in the batch
    for x_input, label in batch: # Batch is 1
        """
        x_input: list of numpy arrays
        label: list of integers with 0 for spoof, 1 bonafide
        """
        for index, (x) in enumerate(x_input):
            # Ensure x is a Tensor
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)

            # Pad each sample for each view
            for view in views:
                view_length = view * sample_rate

                x_view = pad(x.numpy(), padding_type=view_padding_configs[str(view)]['padding_type'],
                             max_len=view_length, random_start=view_padding_configs[str(view)]['random_start'])

                # Convert padded output to Tensor
                if not torch.is_tensor(x_view):
                    x_view = torch.tensor(x_view, dtype=torch.float32)

                # Ensure label is a tensor (fixing the warning)
                if not isinstance(label[index], torch.Tensor):
                    label_tensor = torch.tensor(label[index], dtype=torch.long)
                else:
                    label_tensor = label[index].clone().detach()

                view_batches[view].append((x_view, label_tensor))

    # Convert lists to tensors
    for view in views:
        sequences, labels = zip(*view_batches[view])
        # Ensuring all sequences are tensors
        padded_sequences = torch.stack(sequences)
        labels = torch.stack(labels)  # Convert labels to tensor properly
        view_batches[view] = (padded_sequences, labels)

    # print(view_batches)
    # import sys
    # sys.exit()
    
    return view_batches


def variable_multi_view_collate_fn(batch, top_k=4, min_duration=16000, max_duration=64000, sample_rate=16000, padding_type='zero', random_start=True):
    '''
    Collate function to pad each sample in a batch to multiple views with variable duration
    :param batch: list of tuples (x, label)
    :param top_k: number of views to pad each sample to
    :param min_duration: minimum duration of the audio
    :param max_duration: maximum duration of the audio
    :param sample_rate: sample rate of the audio
    :param padding_type: padding type to use
    :param random_start: whether to randomly start the sample
    :return: dictionary with keys as views and values as tuples of padded sequences and labels

    Example:
    batch = [([1, 2, 3], 0), ([1, 2, 3, 4], 1)]
    variable_multi_view_collate_fn(batch, top_k=2, min_duration=16000, max_duration=32000, sample_rate=16000)
    Output:
    {
        1: (tensor([[1, 2, 3], [1, 2, 3, 4]]), tensor([0, 1])),
        2: (tensor([[1, 2, 3, 0], [1, 2, 3, 4]]), tensor([0, 1]))
    }
    '''
    # Duration of each view should be picked from a range of min_duration to max_duration by a uniform distribution
    # Duration in seconds for each view
    durations = np.random.uniform(
        min_duration, max_duration, top_k).astype(int)
    # Ensure unique durations to avoid key collisions
    views = np.unique(durations)
    view_batches = {view: [] for view in views}
    # Process each sample in the batch
    has_source = len(batch[0]) == 3 if batch else False
    for sample in batch:
        if has_source:
            x, label, source = sample
        else:
            x, label = sample
            source = None
        # Pad each sample for each view
        for view in views:
            view_length = view
            x_view = pad(x, padding_type=padding_type,
                         max_len=view_length, random_start=random_start)
            # Check if x_view is Tensor or numpy array and convert to Tensor if necessary
            if not torch.is_tensor(x_view):
                x_view = torch.from_numpy(x_view)
            if has_source:
                view_batches[view].append((x_view, label, source))
            else:
                view_batches[view].append((x_view, label))

    # Convert lists to tensors
    for view in views:
        if has_source:
            sequences, labels, sources = zip(*view_batches[view])
            padded_sequences = torch.stack(sequences)
            labels = torch.tensor(labels, dtype=torch.long)
            sources = torch.tensor(sources, dtype=torch.long)
            view_batches[view] = (padded_sequences, labels, sources)
        else:
            sequences, labels = zip(*view_batches[view])
            padded_sequences = torch.stack(sequences)
            labels = torch.tensor(labels, dtype=torch.long)
            view_batches[view] = (padded_sequences, labels)

    return view_batches


def mdt_collate_fn(batch, views=[1, 2, 3, 4], sample_rate=16000, padding_type='zero', num_per_sample=4):
    '''
    Collate function to pad each sample in a batch to multiple views
    :param batch: list of tuples (x, label)
    :param views: list of views to pad each sample to
    :param sample_rate: sample rate of the audio
    :param padding_type: padding type to use
    :param max_duration_per_view: maximum duration of the audio per view. 

    :return: dictionary with keys as views and values as tuples of padded sequences and labels

    Example:
    batch = [([1, 2, 3], 0), ([1, 2, 3, 4], 1)]
    multi_view_collate_fn(batch, views=[1, 2], sample_rate=16000)
    Output:
    {
        1: (tensor([[1, 2, 3], [1, 2, 3, 4]]), tensor([0, 1])),
        2: (tensor([[1, 2, 3, 0], [1, 2, 3, 4]]), tensor([0, 1]))
    }
    '''
    view_batches = {view: [] for view in views}

    # Process each sample in the batch
    for x, label in batch:
        # Pad each sample for each view
        for view in views:
            view_length = view * sample_rate
            x_view = pad(x, padding_type=padding_type,  max_len=view_length)
            # Check if x_view is Tensor or numpy array and convert to Tensor if necessary
            if not torch.is_tensor(x_view):
                x_view = torch.from_numpy(x_view)
            view_batches[view].append((x_view, label))

    # Convert lists to tensors
    for view in views:
        sequences, labels = zip(*view_batches[view])
        padded_sequences = torch.stack(sequences)
        labels = torch.tensor(labels, dtype=torch.long)
        view_batches[view] = (padded_sequences, labels)

    return view_batches


class ChunkingCollator(object):
    def __init__(self, **params):
        self.enable_chunking = params.get('enable_chunking', False)
        print("🐍 File: components/collate_fn.py | Line: 146 | __init__ ~ self.enable_chunking",
              self.enable_chunking)
        self.chunk_size = params.get('chunk_size', 64600)
        print("🐍 File: components/collate_fn.py | Line: 149 | __init__ ~ self.chunk_size", self.chunk_size)
        self.overlap_size = params.get(
            'overlap_size', 0)  # Default overlap size is 0
        print("🐍 File: components/collate_fn.py | Line: 153 | __init__ ~ self.overlap_size", self.overlap_size)

    def __call__(self, batch):
        if self.enable_chunking:
            return self.chunking(batch)
        return batch

    def chunking(self, batch):
        chunk_size = self.chunk_size
        overlap_size = self.overlap_size
        step_size = chunk_size - overlap_size

        split_data = []

        for x_inp, utt_id in batch:
            # Calculate number of chunks with overlap
            num_chunks = (len(x_inp) - overlap_size) // step_size

            # handle case where the utterance is smaller than overlap_size
            if num_chunks <= 0:
                padded_chunk = pad(
                    x=x_inp, padding_type='repeat', max_len=chunk_size)
                padded_chunk = Tensor(padded_chunk)
                chunk_id = f"{utt_id}___0"
                split_data.append((padded_chunk, chunk_id))
                continue

            for i in range(num_chunks):
                start = i * step_size
                end = start + chunk_size
                chunk = x_inp[start:end]
                chunk_id = f"{utt_id}___{i+1}"
                split_data.append((chunk, chunk_id))

            # Handle the case where the utterance is smaller than chunk_size
            if num_chunks * step_size + overlap_size < len(x_inp):
                start = num_chunks * step_size
                chunk = x_inp[start:]
                padded_chunk = pad(
                    x=chunk, padding_type='repeat', max_len=chunk_size)
                padded_chunk = Tensor(padded_chunk)
                chunk_id = f"{utt_id}___{num_chunks+1}"
                split_data.append((padded_chunk, chunk_id))

        # Convert to tensors (if they are not already tensors)
        x_inp_list, utt_id_list = zip(*split_data)

        x_inp_tensor = torch.stack(x_inp_list) if isinstance(
            x_inp_list[0], torch.Tensor) else torch.tensor(x_inp_list)
        return x_inp_tensor, utt_id_list


def apply_gpu_augmentation_to_batch(
    batch_tensor: torch.Tensor,
    augmentation_methods: List[str],
    args: Dict[str, Any],
    sample_rate: int = 16000,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Apply augmentation to a batch of audio tensors on GPU.
    Uses torch-audiomentations which supports batch processing for better GPU utilization.
    
    Args:
        batch_tensor: Tensor of shape (batch_size, samples) or (batch_size, channels, samples)
        augmentation_methods: List of augmentation method names
        args: Dictionary containing augmentation configuration
        sample_rate: Sample rate of the audio
        device: Device to run augmentation on (default: cuda if available)
        
    Returns:
        Augmented batch tensor on the same device
    """
    if not augmentation_methods or len(augmentation_methods) == 0:
        return batch_tensor
    

    # Ensure batch_tensor is 3D: (batch_size, channels, samples) for torch-audiomentations
    original_shape = batch_tensor.shape
    if batch_tensor.ndim == 2:
        batch_tensor = batch_tensor.unsqueeze(1)  # (batch_size, 1, samples)
    elif batch_tensor.ndim == 1:
        batch_tensor = batch_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
    
    # Randomly select an augmentation method
    aug_method_name = random.choice(augmentation_methods)
    
    # Import torch-audiomentations directly for batch processing
    try:
        from torch_audiomentations import LowPassFilter
        
        if aug_method_name == 'low_pass_filter':
            # Create augmentation configuration
            aug_config = {
                'min_cutoff_freq': args.get('min_cutoff_freq', 150.0),
                'max_cutoff_freq': args.get('max_cutoff_freq', 7500.0),
                'p': args.get('p', 1.0),
                'sample_rate': sample_rate,
                'output_type': 'dict',
                #'target_rate': args.get('target_rate', sample_rate),
            }
            #print(f"🐍 File: components/collate_fn.py | Line: 542 | apply_gpu_augmentation_to_batch ~ aug_config", aug_config)
            # Create augmentation instance and move to device
            augment = LowPassFilter(**aug_config)
            # .to(device) is safe here because device will be CPU in workers
            #augment = augment.to(device)
            
            # Apply augmentation to entire batch at once (much faster!)
            result = augment(samples=batch_tensor, sample_rate=sample_rate)
            
            # Handle output
            if isinstance(result, dict):
                batch_tensor = result['samples']
            elif isinstance(result, torch.Tensor):
                batch_tensor = result
            else:
                batch_tensor = result.samples
        elif aug_method_name == 'resample':
            # This method should be implemented with torchaudio.transforms.Resample
            pass
            
    except ImportError as e:
        print(f"Warning: Could not import torch-audiomentations: {e}")
        pass
    
    # Restore original shape (remove channel dimension if it was added)
    if len(original_shape) == 2 and batch_tensor.shape[1] == 1:
        batch_tensor = batch_tensor.squeeze(1)  # (batch_size, samples)
    
    return batch_tensor


def multi_view_collate_fn_with_gpu_augmentation(
    batch,
    views=[1, 2, 3, 4],
    sample_rate=16000,
    padding_type='repeat',
    random_start=False,
    view_padding_configs: Dict[str, Dict[str, bool]] = None,
    augmentation_methods: Optional[List[str]] = None,
    args: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None
):
    """
    Multi-view collate function with GPU augmentation support.
    
    This function first creates batches using multi_view_collate_fn,
    then applies augmentation on GPU to each view batch.
    
    Args:
        batch: list of tuples (x, label)
        views: list of views to pad each sample to
        sample_rate: sample rate of the audio
        padding_type: padding type to use
        random_start: whether to randomly start the sample
        view_padding_configs: dictionary of view-specific padding configurations
        augmentation_methods: list of augmentation method names (e.g., ['low_pass_filter'])
        args: dictionary containing augmentation configuration
        device: device to run augmentation on (default: cuda if available)
        
    Returns:
        dictionary with keys as views and values as tuples of (augmented_padded_sequences, labels)
    """
    # First, create batches using the original collate function
    view_batches = multi_view_collate_fn(
        batch, views, sample_rate, padding_type, random_start, view_padding_configs
    )
    
    # Apply GPU augmentation if augmentation_methods are provided
    if augmentation_methods and len(augmentation_methods) > 0 and args is not None:
        # Determine device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Apply augmentation to each view batch
        for view in views:
            sequences, labels = view_batches[view]
            # sequences shape: (batch_size, samples)
            
            # Apply augmentation on GPU
            augmented_sequences = apply_gpu_augmentation_to_batch(
                sequences,
                augmentation_methods,
                args,
                sample_rate,
                device
            )
            
            view_batches[view] = (augmented_sequences, labels)
    
    return view_batches


from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F

try:
    import torchaudio
    from torchaudio.functional import lowpass_biquad
except ImportError:
    torchaudio = None
    lowpass_biquad = None


def _to_tensor(x):
    if torch.is_tensor(x):
        return x.float()
    return torch.tensor(x, dtype=torch.float32)


def _fix_length(
    x: torch.Tensor,
    target_len: int,
    padding_type: str = "repeat",
    random_start: bool = False,
) -> torch.Tensor:
    """
    Make waveform length exactly target_len.
    x: shape [T]
    """
    x = x.flatten()

    if x.numel() == target_len:
        return x

    if x.numel() > target_len:
        if random_start:
            max_start = x.numel() - target_len
            start = torch.randint(0, max_start + 1, (1,)).item()
        else:
            start = 0
        return x[start:start + target_len]

    pad_len = target_len - x.numel()

    if padding_type == "zero":
        return F.pad(x, (0, pad_len))

    if padding_type == "repeat":
        n_repeat = (target_len + x.numel() - 1) // x.numel()
        x_rep = x.repeat(n_repeat)
        return x_rep[:target_len]

    raise ValueError(f"Unsupported padding_type: {padding_type}")


def _resample_waveform(x: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    if orig_sr == new_sr:
        return x
    if torchaudio is None:
        raise ImportError("torchaudio is required for resampling.")
    return torchaudio.functional.resample(x, orig_freq=orig_sr, new_freq=new_sr)


def _apply_band_transform(
    x: torch.Tensor,
    band: str,
    sample_rate: int = 16000,
    wideband_cutoff: int = 7800,
) -> torch.Tensor:
    """
    Apply MBCT band transform to a 1D waveform.
    Input and output are both [T] at sample_rate.
    """
    x = x.flatten()

    if band == "normal":
        return x

    elif band == "narrowband":
        # Simulate bandwidth reduction via 16k -> 8k -> 16k
        x_8k = _resample_waveform(x, orig_sr=sample_rate, new_sr=8000)
        x_16k = _resample_waveform(x_8k, orig_sr=8000, new_sr=sample_rate)
        return x_16k

    elif band == "wideband":
        if lowpass_biquad is None:
            raise ImportError("torchaudio is required for low-pass filtering.")
        # Slightly below Nyquist is safer than exactly 8k at 16k SR
        return lowpass_biquad(x.unsqueeze(0), sample_rate=sample_rate, cutoff_freq=wideband_cutoff).squeeze(0)

    else:
        raise ValueError(f"Unsupported band type: {band}")


def _mbct_apply_configured_band(
    x: torch.Tensor,
    band_name: str,
    cfg: Dict[str, Any],
    sample_rate: int,
) -> torch.Tensor:
    """Apply MBCT band path from ``band_configs`` entry (same rules as :func:`mbct_collate_fn`)."""
    band_type = cfg.get("type", band_name)
    if band_type == "normal":
        return _apply_band_transform(x, band="normal", sample_rate=sample_rate)
    if band_type == "narrowband":
        return _apply_band_transform(x, band="narrowband", sample_rate=sample_rate)
    if band_type == "wideband":
        return _apply_band_transform(
            x,
            band="wideband",
            sample_rate=sample_rate,
            wideband_cutoff=cfg.get("cutoff_hz", 7800),
        )
    raise ValueError(f"Unknown band config type: {band_type}")


def mbct_mdt_collate_fn(
    batch: List[Tuple[torch.Tensor, int]],
    sample_rate: int = 16000,
    max_length_sec: Optional[float] = None,
    padding_type: str = "repeat",
    random_start: bool = False,
    view_padding_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    band_configs: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Multi-duration (MDT) × multi-band (MBCT) collate.

    For each sample, optionally fixes length, then for each duration view pads/crops to
    ``view * sample_rate`` samples, then applies each MBCT band transform. Batch keys are
    ``"{view_key}_{band_name}"`` (e.g. ``1_normal``), matching :class:`MDTLitModule` /
    :class:`MBCTLitModule` flat ``weighted_views`` keys.
    """
    if band_configs is None:
        band_configs = {
            "normal": {"type": "normal"},
            "narrowband": {"type": "narrowband"},
            "wideband": {"type": "wideband", "cutoff_hz": 7800},
        }
    if view_padding_configs is None:
        view_padding_configs = {
            str(i): {"padding_type": "repeat", "random_start": False}
            for i in range(1, 5)
        }

    view_keys = sorted(view_padding_configs.keys(), key=lambda k: int(k))
    composite_keys = [f"{vk}_{bn}" for vk in view_keys for bn in band_configs.keys()]
    view_batches = {k: [] for k in composite_keys}

    target_len = None
    if max_length_sec is not None:
        target_len = int(max_length_sec * sample_rate)

    for x, label in batch:
        x = _to_tensor(x).flatten()
        if target_len is not None:
            x = _fix_length(
                x,
                target_len=target_len,
                padding_type=padding_type,
                random_start=random_start,
            )

        for vk in view_keys:
            vpc = view_padding_configs[vk]
            view_len = int(vk) * sample_rate
            x_pad = pad_tensor(
                x,
                padding_type=vpc["padding_type"],
                max_len=view_len,
                random_start=vpc["random_start"],
            )
            for band_name, cfg in band_configs.items():
                x_band = _mbct_apply_configured_band(
                    x_pad, band_name, cfg, sample_rate
                )
                ck = f"{vk}_{band_name}"
                view_batches[ck].append((x_band, label))

    for ck in view_batches:
        sequences, labels = zip(*view_batches[ck])
        sequences = torch.stack(sequences, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        view_batches[ck] = (sequences, labels)

    return view_batches


def mbct_collate_fn(
    batch: List[Tuple[torch.Tensor, int]],
    sample_rate: int = 16000,
    max_length_sec: Optional[float] = None,
    padding_type: str = "repeat",
    random_start: bool = False,
    band_configs: Optional[Dict[str, Dict]] = None,
):
    """
    Multi-Band Consistency Training collate function.

    Args:
        batch:
            List of (waveform, label), where waveform is 1D audio.
        sample_rate:
            Target sample rate expected by the model, usually 16000.
        max_length_sec:
            If provided, crop/pad all waveforms to fixed duration before band transform.
        padding_type:
            'repeat' or 'zero'.
        random_start:
            Whether to use random crop when waveform is longer than target length.
        band_configs:
            Dict defining band conditions.

    Returns:
        Dict like:
        {
            "normal": (Tensor[B, T], Tensor[B]),
            "narrowband": (Tensor[B, T], Tensor[B]),
            "wideband": (Tensor[B, T], Tensor[B]),
        }
    """
    if band_configs is None:
        band_configs = {
            "normal": {"type": "normal"},
            "narrowband": {"type": "narrowband"},
            "wideband": {"type": "wideband", "cutoff_hz": 7800},
        }

    view_batches = {band_name: [] for band_name in band_configs.keys()}

    target_len = None
    if max_length_sec is not None:
        target_len = int(max_length_sec * sample_rate)

    for x, label in batch:
        x = _to_tensor(x).flatten()

        if target_len is not None:
            x = _fix_length(
                x,
                target_len=target_len,
                padding_type=padding_type,
                random_start=random_start,
            )

        for band_name, cfg in band_configs.items():
            x_view = _mbct_apply_configured_band(x, band_name, cfg, sample_rate)
            view_batches[band_name].append((x_view, label))

    for band_name in view_batches:
        sequences, labels = zip(*view_batches[band_name])
        sequences = torch.stack(sequences, dim=0)   # [B, T]
        labels = torch.tensor(labels, dtype=torch.long)
        view_batches[band_name] = (sequences, labels)

    return view_batches
