import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
def process_data(file_path, max_len, PAD_TOKEN=0, code_dim=4):

    processed_data = []
    with h5py.File(file_path, 'r') as f:
        # 1. 提取所有 history (变长一维数组，每 code_dim 个整数为一个 item 的语义码)
        # 2. 提取所有 target (shape: (N, code_dim) 的语义码数组)
        histories = f['history'][:]
        targets = f['target'][:]

    pad_code = [PAD_TOKEN] * code_dim
    # 将数据打包成字典格式，history 还原为 list of code lists
    for h, t in zip(histories, targets):
        processed_data.append({
            'history': h.reshape(-1, code_dim).tolist(),  # list of [c0,c1,c2,c3]
            'target': t.tolist()                          # [c0,c1,c2,c3]
        })

    # Apply padding or truncation
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
    
def item2code(code_path, codebook_size):
    """
    Convert itemID to code
    :param code_path: npy file path to store rqvae codes
    :return: dict item_to_code, code_to_item
    """
    data = np.load(code_path, allow_pickle=True)
    item_to_code = {}
    code_to_item = {}
    
    # for index, code in enumerate(data):
    #     item_to_code[index + 1] = code
    #     code_to_item[tuple(code)] = index + 1
    for index, code in enumerate(data):
        offsets = [c + i * codebook_size + 1 for i,c in enumerate(code)]
        item_to_code[index + 1] = offsets
        code_to_item[tuple(offsets)] = index + 1

    return item_to_code, code_to_item

class GenRecDataset(Dataset):
    def __init__(self, dataset_path, code_path, max_len, PAD_TOKEN=0, codebook_size=8, prof_h5_paths=None):
        """
        Initialize the GenRecDataset.
        Args:
            dataset_path (str): Path to the dataset file.
            code_path (str): Path to the item-to-code mapping file.
            max_len (int): Maximum length for padding or truncation.
            PAD_TOKEN (int, optional): Token used for padding. Defaults to 0.
            prof_h5_paths (dict, optional): Dict mapping level names to H5 file paths,
                e.g. {'prof_lvl1': '...h5', 'prof_lvl2': '...h5', 'prof_lvl3': '...h5'}.
        """
        self.dataset_path = dataset_path
        self.code_path = code_path
        self.codebook_size = codebook_size
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        self.prof_h5_paths = prof_h5_paths
        # Load item-to-code mapping
        self.item_to_code, self.code_to_item = item2code(code_path, codebook_size)
        # Process the dataset
        self.data = self._prepare_data()
        # Load professional BERT embeddings: {level: np.ndarray (N, 5, 768)}
        self.prof_embeddings = self._load_prof_embeddings()
        
    def _load_prof_embeddings(self):
        """
        Load professional hierarchy BERT embeddings from H5 files.
        Returns:
            dict or None: {level_name: np.ndarray of shape (N, 5, 768)}
        """
        if self.prof_h5_paths is None:
            return None
        prof_data = {}
        for level, path in self.prof_h5_paths.items():
            with h5py.File(path, 'r') as f:
                key = list(f.keys())[0]
                prof_data[level] = f[key][:]  # (N, 5, 768)
        return prof_data

    def _prepare_data(self):
        """
        Process the dataset and convert items to codes.
        Returns:
            list: Processed data with items converted to codes.
        """
        # Process the data using the process_data function
        processed_data = process_data(
            self.dataset_path, self.max_len, self.PAD_TOKEN
        )
        # history and target are already stored as semantic codes in the H5 file
        return processed_data
    
    def __getitem__(self, index):
        """
        Get a single data item by index.
        Args:
            index (int): Index of the data item.
        Returns:
            dict: A dictionary containing 'history', 'target', and optionally
                  'prof_lvl1', 'prof_lvl2', 'prof_lvl3' of shape (5, 768).
        """
        item = dict(self.data[index])
        if self.prof_embeddings is not None:
            item['prof_lvl1'] = self.prof_embeddings['prof_lvl1'][index]  # (5, 768)
            item['prof_lvl2'] = self.prof_embeddings['prof_lvl2'][index]  # (5, 768)
            item['prof_lvl3'] = self.prof_embeddings['prof_lvl3'][index]  # (5, 768)
        return item
    
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
        """
        crate attention mask for input sequence.
        
        Args:
            batch (list): List of samples from the dataset.
        
        Returns:
            dict: Batched data with padded sequences.
        """
        # Assuming each item in batch is a dictionary with 'history' and 'target'
        histories = [item['history'] for item in batch]
        targets = [item['target'] for item in batch]

        # Flatten histories and targets
        flattened_histories = torch.stack(
            [torch.tensor([elem for sublist in history for elem in sublist], dtype=torch.int64) for history in histories]
        )
        flattened_targets = torch.stack(
            [torch.tensor(target, dtype=torch.int64) for target in targets]
        )

        # Create attention masks for flattened histories
        attention_masks = torch.stack(
            [torch.tensor([1 if elem != pad_token else 0 for elem in h], dtype=torch.int64) for h in flattened_histories]
        )

        result = {'history': flattened_histories, 'target': flattened_targets, 'attention_mask': attention_masks}

        result['prof_lvl1'] = torch.stack([torch.tensor(item['prof_lvl1'], dtype=torch.float32) for item in batch])  # (B, 5, 768)
        result['prof_lvl2'] = torch.stack([torch.tensor(item['prof_lvl2'], dtype=torch.float32) for item in batch])  # (B, 5, 768)
        result['prof_lvl3'] = torch.stack([torch.tensor(item['prof_lvl3'], dtype=torch.float32) for item in batch])  # (B, 5, 768)

        return result
