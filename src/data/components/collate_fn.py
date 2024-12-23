import torch
from src.data.components.dataio import pad
import numpy as np
from torch import Tensor


def multi_view_collate_fn(batch, views=[1, 2, 3, 4], sample_rate=16000, padding_type='zero', random_start=True):
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
    view_batches = {view: [] for view in views}

    # Process each sample in the batch
    for x, label in batch:
        # Pad each sample for each view
        for view in views:
            view_length = view * sample_rate
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
