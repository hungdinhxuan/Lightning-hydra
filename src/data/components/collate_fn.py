import torch
from src.data.components.dataio import pad

def multi_view_collate_fn(batch, views=[1, 2, 3, 4], sample_rate=16000, padding_type='zero', random_start=True):
    view_batches = {view: [] for view in views}

    # Process each sample in the batch
    for x, label in batch:
        # Pad each sample for each view
        for view in views:
            view_length = view * sample_rate
            x_view = pad(x, padding_type=padding_type, max_len=view_length, random_start=random_start)
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