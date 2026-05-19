import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
def process_data(file_path, max_len, PAD_TOKEN=0, code_dim=4):

    processed_data = []
    with h5py.File(file_path, 'r') as f:
        histories = f['history'][:]
        targets = f['target'][:]
        user_ids = f['user_id'][:] if 'user_id' in f else None

    pad_code = [PAD_TOKEN] * code_dim
    if user_ids is not None:
        for uid, h, t in zip(user_ids, histories, targets):
            processed_data.append({
                'user_id': int(uid),
                'history': h.reshape(-1, code_dim).tolist(),
                'target': t.reshape(-1, code_dim).tolist()    # list of [c0,c1,c2,c3]
            })
    else:
        for h, t in zip(histories, targets):
            processed_data.append({
                'history': h.reshape(-1, code_dim).tolist(),
                'target': t.reshape(-1, code_dim).tolist()
            })

    for item in processed_data:
        item['history'] = pad_or_truncate(item['history'], max_len, pad_code)

    return processed_data

def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    """
    Pad or truncate a sequence to a specified maximum length.

    Args:
        sequence (list): Input sequence.
        max_len (int): Maximum length for the sequence.
        PAD_TOKEN: Padding value; can be an int or a list (e.g. [0,0,0,0] for code padding).

    Returns:
        list: Padded or truncated sequence.
    """
    if len(sequence) > max_len:
        # Truncate sequence
        return sequence[-max_len:]
    else:
        # Left pad sequence with PAD_TOKEN
        pad_len = max_len - len(sequence)
        if isinstance(PAD_TOKEN, list):
            padding = [list(PAD_TOKEN) for _ in range(pad_len)]
        else:
            padding = [PAD_TOKEN] * pad_len
        return padding + sequence
    
class GenRecDataset(Dataset):
    def __init__(self, dataset_path, max_len, PAD_TOKEN=0):
        """
        Initialize the GenRecDataset.
        Args:
            dataset_path (str): Path to the dataset file.
            max_len (int): Maximum length for padding or truncation.
            PAD_TOKEN (int, optional): Token used for padding. Defaults to 0.
        """
        self.dataset_path = dataset_path
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        self.data = self._prepare_data()
        
    def _prepare_data(self):
        """
        Process the dataset and convert items to codes.
        Returns:
            list: Processed data with items converted to codes.
        """
        # history and target are already stored as semantic codes in the H5 file
        processed_data = process_data(
            self.dataset_path, self.max_len, self.PAD_TOKEN
        )
        return processed_data
    
    def __getitem__(self, index):
        """
        Get a single data item by index.
        Args:
            index (int): Index of the data item.
        Returns:
            dict: A dictionary containing 'history' and 'target'.
        """
        return self.data[index]
    
    def __len__(self):
        """
        Get the total number of data.
        Returns:
            int: Total number of data.
        """
        return len(self.data)

# Dataoding
class GenRecDataLoader(DataLoader):
    """
    GenRecDataLoader for Generative Recommendation tasks.
    
    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): Number of subprocesses to use for data loading.
        collate_fn (callable, optional): Function to merge a list of samples to form a mini-batch.
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=2, collate_fn=None):
        collate_fn = self.collate_fn
        super(GenRecDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers, collate_fn=collate_fn)
    
            
    def collate_fn(self, batch, pad_token=0):
        histories = [item['history'] for item in batch]
        targets = [item['target'] for item in batch]

        flattened_histories = torch.stack(
            [torch.tensor([elem for sublist in history for elem in sublist], dtype=torch.int64) for history in histories]
        )

        flat_targets = [
            [c for code_list in t for c in code_list] for t in targets
        ]
        max_target_len = max(len(ft) for ft in flat_targets)
        padded_targets = torch.stack([
            torch.tensor(ft + [-100] * (max_target_len - len(ft)), dtype=torch.int64)
            for ft in flat_targets
        ])

        attention_masks = torch.stack(
            [torch.tensor([1 if elem != pad_token else 0 for elem in h], dtype=torch.int64) for h in flattened_histories]
        )

        return {'history': flattened_histories, 'target': padded_targets, 'attention_mask': attention_masks}
