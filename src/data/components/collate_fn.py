import torch
from src.data.components.dataio import pad
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
    for x, label in batch:
        # Pad each sample for each view
        for view in views:
            view_length = view * sample_rate
            x_view = pad(x, padding_type=view_padding_configs[str(view)]['padding_type'],
                         max_len=view_length, random_start=view_padding_configs[str(view)]['random_start'])
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

# def multi_view_collate_fn_for_scl(batch: List[Tuple[List[np.ndarray], Tensor]],
#                                   views: List[int] = [1, 2, 3, 4],
#                                   sample_rate: int = 16000,
#                                   padding_type: str = 'repeat',
#                                   random_start: bool = False,
#                                   view_padding_configs: Dict[str, Dict[str, bool]] = None) -> Dict[int, Tuple[Tensor, Tensor]]:
#     """
#     Collate function to pad multiple audio samples in a batch to multiple views.

#     Args:
#         batch: List of tuples (batch_data, label) where batch_data is a list of audio samples
#         views: List of views to pad each sample to
#         sample_rate: Sample rate of the audio
#         padding_type: Default padding type if view_padding_configs not provided
#         random_start: Default random start setting if view_padding_configs not provided
#         view_padding_configs: Dictionary of view-specific padding configurations

#     Returns:
#         Dictionary with views as keys and tuples of (padded_sequences, labels) as values
#     """
#     # Set default configurations if none provided
#     if view_padding_configs is None:
#         view_padding_configs = {
#             str(view): {'padding_type': padding_type, 'random_start': random_start}
#             for view in views
#         }

#     # Initialize dictionary to store batches for each view
#     view_batches = {view: {'sequences': [], 'labels': []} for view in views}

#     for batch_data, label in batch:
#         # Process each audio sample in the batch_data list
#         for audio in batch_data:
#             if isinstance(audio, np.ndarray):
#                 audio = torch.from_numpy(audio)

#             # Process for each view
#             for view in views:
#                 view_length = view * sample_rate
#                 config = view_padding_configs[str(view)]

#                 x_padded = nii_wav_aug.batch_pad_for_multiview(
#                     audio, sample_rate, view_length, random_trim_nosil=config['random_start'], repeat_pad=True if config['padding_type'] == 'repeat' else False)
#                 # Use the existing pad function
#                 # x_padded = pad(
#                 #     audio,
#                 #     padding_type=config['padding_type'],
#                 #     max_len=view_length,
#                 #     random_start=config['random_start']
#                 # )
#                 x_padded = np.concatenate(x_padded, axis=1)

#                 if not torch.is_tensor(x_padded):
#                     x_padded = torch.from_numpy(x_padded)

#                 view_batches[view]['sequences'].append(x_padded)
#                 view_batches[view]['labels'].append(label)
#     print(view_batches)
#     # Convert lists to tensors for each view
#     result = {}
#     for view in views:
#         sequences = torch.stack(view_batches[view]['sequences'])
#         # Stack all labels and take only unique values since they're repeated
#         labels = torch.stack(view_batches[view]['labels']).unique(dim=0)
#         result[view] = (sequences, labels)

#     return result
# def multi_view_collate_fn_for_scl(batch, views=[1, 2, 3, 4], sample_rate=16000, padding_type='repeat', random_start=False, view_padding_configs: Dict[str, Dict[str, bool]] = None):
#     '''
#     Collate function to pad each sample in a batch to multiple views
#     :param batch: list of tuples (x, label)
#     :param views: list of views to pad each sample to
#     :param sample_rate: sample rate of the audio
#     :param padding_type: padding type to use
#     :param random_start: whether to randomly start the sample
#     :return: dictionary with keys as views and values as tuples of padded sequences and labels

#     Example:
#     batch = [([1, 2, 3], 0), ([1, 2, 3, 4], 1)]
#     multi_view_collate_fn(batch, views=[1, 2], sample_rate=16000)
#     Output:
#     {
#         1: (tensor([[1, 2, 3], [1, 2, 3, 4]]), tensor([0, 1])),
#         2: (tensor([[1, 2, 3, 0], [1, 2, 3, 4]]), tensor([0, 1]))
#     }
#     '''
#     # Set default configurations if none provided
#     if view_padding_configs is None:
#         view_padding_configs = {
#             str(i): {'padding_type': 'repeat', 'random_start': False}
#             for i in range(1, 5)
#         }

#     # Extract views from config and convert to integers
#     views = [int(view) for view in view_padding_configs]

#     view_batches = {view: [] for view in views}
#     # Warning: padding_type and random_start are not used in this function
#     # print("Warning: padding_type and random_start are not used in this function. Please use view_padding_configs instead")

#     # Process each sample in the batch

#     for index, (x_input, label) in enumerate(batch):
#         """
#         x_input: list of numpy arrays
#         label: list of integers with 0 for spoof, 1 bonafide
#         """
#         for x in x_input:
#             # Check if x is Tensor or numpy array and convert to Tensor if necessary

#             # Pad each sample for each view
#             for view in views:
#                 view_length = view * sample_rate

#                 x_view = pad(x, padding_type=view_padding_configs[str(view)]['padding_type'],
#                              max_len=view_length, random_start=view_padding_configs[str(view)]['random_start'])

#                 if not torch.is_tensor(x_view):
#                     x_view = torch.from_numpy(x_view)

#                 view_batches[view].append((x_view, label[index]))

#     # Convert lists to tensors
#     for view in views:
#         sequences, labels = zip(*view_batches[view])
#         padded_sequences = torch.stack(sequences)
#         view_batches[view] = (padded_sequences, labels)

#     return view_batches

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
    for x, label in batch:
        # Pad each sample for each view
        for view in views:
            view_length = view
            x_view = pad(x, padding_type=padding_type,
                         max_len=view_length, random_start=random_start)
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
        from torch_audiomentations import LowPassFilter, Compose, Resample
        
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
